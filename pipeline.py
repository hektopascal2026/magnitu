"""
ML Pipeline for Magnitu 2.

Two architectures behind the same interface:
- "tfidf":       TF-IDF + Logistic Regression (original Magnitu 1)
- "transformer": Cached DistilRoBERTa embeddings + Logistic Regression (Magnitu 2)

The transformer path computes embeddings once at sync time and stores them in
the DB.  Training and scoring use these cached embeddings with a lightweight
LogReg classifier — so labeling stays snappy.  The TF-IDF path is kept as a
fallback and is used by the recipe distiller for knowledge distillation.
"""
import json
import logging
import struct
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

from typing import Optional, List, Dict

import db
from config import MODELS_DIR, get_config

logger = logging.getLogger(__name__)

CLASSES = ["investigation_lead", "important", "background", "noise"]

CLASS_WEIGHT_MAP = {
    "investigation_lead": 1.0,
    "important": 0.66,
    "background": 0.33,
    "noise": 0.0,
}


# ═══════════════════════════════════════════════════════════════════
#  Embedding helpers (Magnitu 2)
# ═══════════════════════════════════════════════════════════════════

_embedder = None   # lazy-loaded singleton


def _get_embedder():
    """Lazy-load the transformer model + tokenizer. Cached after first call."""
    global _embedder
    if _embedder is not None:
        return _embedder

    import torch
    from transformers import AutoTokenizer, AutoModel

    config = get_config()
    model_name = config.get("transformer_model_name", "distilroberta-base")

    logger.info("Loading transformer model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Select device: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    logger.info("Transformer on device: %s", device)

    _embedder = {"tokenizer": tokenizer, "model": model, "device": device}
    return _embedder


def compute_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Compute [CLS] embeddings for a list of texts using the transformer.
    Returns ndarray of shape (len(texts), embedding_dim).
    """
    import torch

    embedder = _get_embedder()
    tokenizer = embedder["tokenizer"]
    model = embedder["model"]
    device = embedder["device"]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Use [CLS] token embedding (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a 1-D float32 embedding to bytes for SQLite storage."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int = 768) -> np.ndarray:
    """Deserialize bytes back to a 1-D float32 embedding."""
    return np.frombuffer(data, dtype=np.float32)


def embed_entries(entries: List[dict]) -> List[bytes]:
    """Compute embeddings for a list of entry dicts. Returns list of bytes."""
    texts = []
    for e in entries:
        text = "{} {} {}".format(
            e.get("title", ""), e.get("description", ""), e.get("content", "")
        ).strip()
        if not text:
            text = "(empty)"
        texts.append(text)

    embeddings = compute_embeddings(texts)
    return [embedding_to_bytes(emb) for emb in embeddings]


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF pipeline (Magnitu 1 — kept as fallback + recipe distiller)
# ═══════════════════════════════════════════════════════════════════

def _prepare_text(entries: List[dict]) -> pd.DataFrame:
    """Convert entries into a DataFrame with text and structured features."""
    rows = []
    for e in entries:
        text = "{} {} {}".format(
            e.get("title", ""), e.get("description", ""), e.get("content", "")
        ).strip()
        rows.append({
            "text": text,
            "source_type": e.get("source_type", "unknown"),
            "text_length": len(text),
        })
    return pd.DataFrame(rows)


def build_tfidf_pipeline() -> Pipeline:
    """Build the scikit-learn pipeline: TF-IDF + structured features -> LogReg."""
    text_transformer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.95,
    )

    source_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "text"),
            ("source", source_transformer, ["source_type"]),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    return pipeline


# ═══════════════════════════════════════════════════════════════════
#  Unified interface — delegates to the configured architecture
# ═══════════════════════════════════════════════════════════════════

def _get_architecture() -> str:
    """Return the current architecture from config."""
    config = get_config()
    return config.get("model_architecture", "transformer")


def train() -> dict:
    """Train a new model on all labeled entries using the configured architecture."""
    arch = _get_architecture()
    if arch == "transformer":
        return _train_transformer()
    return _train_tfidf()


def load_active_model():
    """Load the currently active model from disk. Returns sklearn pipeline or classifier dict."""
    model_info = db.get_active_model()
    if not model_info or not model_info.get("model_path"):
        return None
    model_path = model_info["model_path"]
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def score_entries(entries: List[dict]) -> List[dict]:
    """Score entries using the active model. Architecture-aware."""
    model_info = db.get_active_model()
    if not model_info:
        return []

    arch = model_info.get("architecture", "tfidf")
    if arch == "transformer":
        return _score_transformer(entries, model_info)
    return _score_tfidf(entries)


def get_feature_importance() -> dict:
    """Get feature importance. For TF-IDF models only (used by recipe distiller)."""
    model_info = db.get_active_model()
    if not model_info:
        return {}

    arch = model_info.get("architecture", "tfidf")

    if arch == "tfidf":
        return _get_tfidf_feature_importance()

    # For transformer models, there's no direct keyword importance.
    # The distiller uses knowledge distillation instead.
    return {}


# ═══════════════════════════════════════════════════════════════════
#  Transformer training + scoring (Magnitu 2)
# ═══════════════════════════════════════════════════════════════════

def _train_transformer() -> dict:
    """Train a LogReg classifier on cached transformer embeddings."""
    config = get_config()
    min_labels = config.get("min_labels_to_train", 20)
    embedding_dim = config.get("embedding_dim", 768)

    labeled = db.get_all_labels()
    if len(labeled) < min_labels:
        return {
            "success": False,
            "error": "Need at least {} labels to train. Currently have {}.".format(
                min_labels, len(labeled)
            ),
            "label_count": len(labeled),
        }

    # Collect embeddings and labels
    conn = db.get_db()
    X_list = []
    y_list = []
    missing_embeddings = []

    for lbl in labeled:
        row = conn.execute(
            "SELECT embedding FROM entries WHERE entry_type = ? AND entry_id = ?",
            (lbl["entry_type"], lbl["entry_id"])
        ).fetchone()
        if row and row["embedding"]:
            emb = bytes_to_embedding(row["embedding"], embedding_dim)
            X_list.append(emb)
            y_list.append(lbl["label"])
        else:
            missing_embeddings.append(lbl)
    conn.close()

    # Compute missing embeddings on the fly
    if missing_embeddings:
        logger.info("Computing %d missing embeddings for training", len(missing_embeddings))
        emb_bytes = embed_entries(missing_embeddings)
        updates = []
        for lbl, eb in zip(missing_embeddings, emb_bytes):
            updates.append((eb, lbl["entry_type"], lbl["entry_id"]))
            emb = bytes_to_embedding(eb, embedding_dim)
            X_list.append(emb)
            y_list.append(lbl["label"])
        db.store_embeddings_batch(updates)

    if len(X_list) < min_labels:
        return {
            "success": False,
            "error": "Not enough entries with embeddings. Try syncing first.",
            "label_count": len(labeled),
        }

    X = np.array(X_list)
    y = y_list
    labels_series = pd.Series(y)
    label_counts = labels_series.value_counts()
    min_class_count = label_counts.min()

    if min_class_count < 2:
        X_train, X_test = X, X
        y_train, y_test = y, y
        split_note = "All data used for training (some classes have <2 samples)"
    else:
        test_size = min(0.2, min_class_count / len(y))
        test_size = max(test_size, 0.1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        split_note = "80/{} train/test split".format(int(test_size * 100))

    # Train LogReg on embeddings
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Save classifier
    version = db.get_next_model_version()
    model_filename = "model_v{}.joblib".format(version)
    model_path = str(MODELS_DIR / model_filename)
    joblib.dump(clf, model_path)

    label_dist = {k: int(v) for k, v in labels_series.value_counts().items()}

    db.save_model_record(
        version=version,
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        label_count=len(labeled),
        label_dist=label_dist,
        feature_count=X.shape[1],
        model_path=model_path,
    )

    # Update architecture in the model record
    conn = db.get_db()
    conn.execute(
        "UPDATE models SET architecture = ? WHERE version = ?",
        ("transformer", version)
    )
    conn.commit()
    conn.close()

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = json.loads(json.dumps(report, default=float))

    return {
        "success": True,
        "version": version,
        "architecture": "transformer",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "label_count": len(labeled),
        "label_distribution": label_dist,
        "feature_count": int(X.shape[1]),
        "model_path": model_path,
        "split_note": split_note,
        "class_report": report,
    }


def _score_transformer(entries: List[dict], model_info: dict) -> List[dict]:
    """Score entries using cached embeddings + LogReg classifier."""
    model_path = model_info.get("model_path")
    if not model_path or not Path(model_path).exists():
        return []

    clf = joblib.load(model_path)
    config = get_config()
    embedding_dim = config.get("embedding_dim", 768)

    # Gather embeddings — use cached where available, compute on the fly otherwise
    conn = db.get_db()
    embeddings = []
    to_compute = []
    to_compute_indices = []

    for i, entry in enumerate(entries):
        row = conn.execute(
            "SELECT embedding FROM entries WHERE entry_type = ? AND entry_id = ?",
            (entry["entry_type"], entry["entry_id"])
        ).fetchone()
        if row and row["embedding"]:
            embeddings.append((i, bytes_to_embedding(row["embedding"], embedding_dim)))
        else:
            to_compute.append(entry)
            to_compute_indices.append(i)
    conn.close()

    # Compute missing embeddings
    if to_compute:
        new_emb_bytes = embed_entries(to_compute)
        new_emb_arrays = [bytes_to_embedding(b, embedding_dim) for b in new_emb_bytes]
        updates = []
        for entry, eb in zip(to_compute, new_emb_bytes):
            updates.append((eb, entry["entry_type"], entry["entry_id"]))
        db.store_embeddings_batch(updates)
        for idx, emb in zip(to_compute_indices, new_emb_arrays):
            embeddings.append((idx, emb))

    # Sort by original index
    embeddings.sort(key=lambda x: x[0])
    X = np.array([emb for _, emb in embeddings])

    if len(X) == 0:
        return []

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    class_names = clf.classes_.tolist()

    results = []
    for i, entry in enumerate(entries):
        probs = dict(zip(class_names, probabilities[i].tolist()))
        composite = sum(probs.get(c, 0) * CLASS_WEIGHT_MAP.get(c, 0) for c in class_names)
        results.append({
            "entry_type": entry["entry_type"],
            "entry_id": entry["entry_id"],
            "relevance_score": round(composite, 4),
            "predicted_label": predictions[i],
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        })

    return results


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF training + scoring (Magnitu 1 fallback)
# ═══════════════════════════════════════════════════════════════════

def _train_tfidf() -> dict:
    """Train using the original TF-IDF + LogReg pipeline."""
    config = get_config()
    min_labels = config.get("min_labels_to_train", 20)

    labeled = db.get_all_labels()
    if len(labeled) < min_labels:
        return {
            "success": False,
            "error": "Need at least {} labels to train. Currently have {}.".format(
                min_labels, len(labeled)
            ),
            "label_count": len(labeled),
        }

    df = _prepare_text(labeled)
    labels = [e["label"] for e in labeled]

    label_counts = pd.Series(labels).value_counts()
    min_class_count = label_counts.min()

    if min_class_count < 2:
        X_train, X_test = df, df
        y_train, y_test = labels, labels
        split_note = "All data used for training (some classes have <2 samples)"
    else:
        test_size = min(0.2, min_class_count / len(labels))
        test_size = max(test_size, 0.1)
        X_train, X_test, y_train, y_test = train_test_split(
            df, labels, test_size=test_size, stratify=labels, random_state=42
        )
        split_note = "80/{} train/test split".format(int(test_size * 100))

    pipeline = build_tfidf_pipeline()

    try:
        pipeline.fit(X_train, y_train)
    except ValueError:
        pipeline.named_steps["features"].transformers[0] = (
            "text",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
                min_df=1,
                max_df=1.0,
            ),
            "text",
        )
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    version = db.get_next_model_version()
    model_filename = "model_v{}.joblib".format(version)
    model_path = str(MODELS_DIR / model_filename)
    joblib.dump(pipeline, model_path)

    tfidf = pipeline.named_steps["features"].transformers_[0][1]
    feature_count = len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else 0

    label_dist = {k: int(v) for k, v in pd.Series(labels).value_counts().items()}

    db.save_model_record(
        version=version,
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        label_count=len(labeled),
        label_dist=label_dist,
        feature_count=feature_count,
        model_path=model_path,
    )

    # Mark architecture
    conn = db.get_db()
    conn.execute(
        "UPDATE models SET architecture = ? WHERE version = ?",
        ("tfidf", version)
    )
    conn.commit()
    conn.close()

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = json.loads(json.dumps(report, default=float))

    return {
        "success": True,
        "version": version,
        "architecture": "tfidf",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "label_count": len(labeled),
        "label_distribution": label_dist,
        "feature_count": int(feature_count),
        "model_path": model_path,
        "split_note": split_note,
        "class_report": report,
    }


def _score_tfidf(entries: List[dict]) -> List[dict]:
    """Score entries using the TF-IDF + LogReg pipeline."""
    model = load_active_model()
    if model is None:
        return []

    df = _prepare_text(entries)
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    class_names = model.classes_.tolist()

    results = []
    for i, entry in enumerate(entries):
        probs = dict(zip(class_names, probabilities[i].tolist()))
        composite = sum(probs.get(c, 0) * CLASS_WEIGHT_MAP.get(c, 0) for c in class_names)
        results.append({
            "entry_type": entry["entry_type"],
            "entry_id": entry["entry_id"],
            "relevance_score": round(composite, 4),
            "predicted_label": predictions[i],
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        })

    return results


def _get_tfidf_feature_importance() -> dict:
    """Get feature coefficients from the active TF-IDF model."""
    model = load_active_model()
    if model is None:
        return {}

    # Check if this is actually a TF-IDF pipeline (has 'features' step)
    if not hasattr(model, "named_steps"):
        return {}

    preprocessor = model.named_steps.get("features")
    if preprocessor is None:
        return {}

    tfidf = preprocessor.transformers_[0][1]
    tfidf_names = tfidf.get_feature_names_out().tolist()

    try:
        source_enc = preprocessor.transformers_[1][1]
        source_names = source_enc.get_feature_names_out().tolist()
    except (IndexError, AttributeError):
        source_names = []

    all_names = tfidf_names + source_names
    classifier = model.named_steps["classifier"]
    class_names = classifier.classes_.tolist()
    coef_matrix = classifier.coef_

    result = {}
    for i, cls in enumerate(class_names):
        coefs = coef_matrix[i]
        pairs = list(zip(all_names[:len(coefs)], coefs.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        result[cls] = pairs

    return result


# ═══════════════════════════════════════════════════════════════════
#  Knowledge distillation: train a TF-IDF student from transformer scores
#  (used by the recipe distiller to produce keyword-weight recipes)
# ═══════════════════════════════════════════════════════════════════

def train_tfidf_student() -> Optional[Pipeline]:
    """
    Train a TF-IDF + LogReg 'student' model that learns from the transformer
    model's predictions on ALL entries (not just labeled ones).

    The student captures the transformer's knowledge in a form that can be
    distilled into a keyword recipe for seismo's PHP.

    Returns the trained student pipeline, or None if not possible.
    """
    model_info = db.get_active_model()
    if not model_info or model_info.get("architecture") != "transformer":
        return None

    # Score all entries with the transformer model
    all_entries = db.get_all_entries()
    if len(all_entries) < 20:
        return None

    scores = score_entries(all_entries)
    if not scores:
        return None

    # Build training data: entry text → transformer's predicted label
    df = _prepare_text(all_entries)
    teacher_labels = [s["predicted_label"] for s in scores]

    # Augment with human labels (they take precedence)
    human_labels = {
        (lbl["entry_type"], lbl["entry_id"]): lbl["label"]
        for lbl in db.get_all_labels()
    }
    for i, entry in enumerate(all_entries):
        key = (entry["entry_type"], entry["entry_id"])
        if key in human_labels:
            teacher_labels[i] = human_labels[key]

    # Build and train the student pipeline
    student = build_tfidf_pipeline()

    try:
        student.fit(df, teacher_labels)
    except ValueError:
        student.named_steps["features"].transformers[0] = (
            "text",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
                min_df=1,
                max_df=1.0,
            ),
            "text",
        )
        student.fit(df, teacher_labels)

    return student
