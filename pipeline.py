"""
ML Pipeline: TF-IDF + Logistic Regression for entry relevance classification.
Handles feature extraction, training, scoring, and model persistence.
"""
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from typing import Optional, List, Dict

import db
from config import MODELS_DIR, get_config

CLASSES = ["investigation_lead", "important", "background", "noise"]


def _prepare_text(entries: list[dict]) -> pd.DataFrame:
    """Convert entries into a DataFrame with text and structured features."""
    rows = []
    for e in entries:
        text = f"{e.get('title', '')} {e.get('description', '')} {e.get('content', '')}"
        text = text.strip()
        rows.append({
            "text": text,
            "source_type": e.get("source_type", "unknown"),
            "text_length": len(text),
        })
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
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
            multi_class="multinomial",
            random_state=42,
        )),
    ])

    return pipeline


def train() -> dict:
    """
    Train a new model on all labeled entries.
    Returns a dict with model info and metrics.
    """
    config = get_config()
    min_labels = config.get("min_labels_to_train", 20)

    # Get all labeled entries
    labeled = db.get_all_labels()
    if len(labeled) < min_labels:
        return {
            "success": False,
            "error": f"Need at least {min_labels} labels to train. Currently have {len(labeled)}.",
            "label_count": len(labeled),
        }

    # Prepare data
    df = _prepare_text(labeled)
    labels = [e["label"] for e in labeled]

    # Handle case where some classes have too few samples for stratified split
    label_counts = pd.Series(labels).value_counts()
    min_class_count = label_counts.min()

    if min_class_count < 2:
        # If any class has fewer than 2 samples, use all data for training (no validation split)
        X_train, X_test = df, df
        y_train, y_test = labels, labels
        split_note = "All data used for training (some classes have <2 samples)"
    else:
        test_size = min(0.2, min_class_count / len(labels))
        test_size = max(test_size, 0.1)
        X_train, X_test, y_train, y_test = train_test_split(
            df, labels, test_size=test_size, stratify=labels, random_state=42
        )
        split_note = f"80/{int(test_size*100)} train/test split"

    # Build and train pipeline
    pipeline = build_pipeline()

    # If min_df=2 fails with small datasets, relax it
    try:
        pipeline.fit(X_train, y_train)
    except ValueError:
        # Relax TF-IDF constraints for small datasets
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

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Save model artifact
    version = db.get_next_model_version()
    model_filename = f"model_v{version}.joblib"
    model_path = str(MODELS_DIR / model_filename)
    joblib.dump(pipeline, model_path)

    # Get feature count
    tfidf = pipeline.named_steps["features"].transformers_[0][1]
    feature_count = len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else 0

    # Label distribution (convert numpy.int64 -> int for JSON serialization)
    label_dist = {k: int(v) for k, v in pd.Series(labels).value_counts().items()}

    # Save model record
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

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    # Convert numpy floats in report to native Python types for JSON serialization
    report = json.loads(json.dumps(report, default=float))

    return {
        "success": True,
        "version": version,
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


def load_active_model() -> Optional[Pipeline]:
    """Load the currently active model from disk."""
    model_info = db.get_active_model()
    if not model_info or not model_info.get("model_path"):
        return None
    model_path = model_info["model_path"]
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def score_entries(entries: list[dict]) -> list[dict]:
    """
    Score entries using the active model.
    Returns list of {entry_type, entry_id, relevance_score, predicted_label, probabilities}.
    """
    model = load_active_model()
    if model is None:
        return []

    df = _prepare_text(entries)
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    class_names = model.classes_.tolist()

    # Class weights for composite score
    class_weight_map = {
        "investigation_lead": 1.0,
        "important": 0.66,
        "background": 0.33,
        "noise": 0.0,
    }

    results = []
    for i, entry in enumerate(entries):
        probs = dict(zip(class_names, probabilities[i].tolist()))
        composite = sum(probs.get(c, 0) * class_weight_map.get(c, 0) for c in class_names)

        results.append({
            "entry_type": entry["entry_type"],
            "entry_id": entry["entry_id"],
            "relevance_score": round(composite, 4),
            "predicted_label": predictions[i],
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        })

    return results


def get_feature_importance() -> dict:
    """
    Get feature coefficients from the active model.
    Returns {class_name: [(feature, weight), ...]} sorted by absolute weight.
    """
    model = load_active_model()
    if model is None:
        return {}

    # Get TF-IDF feature names
    preprocessor = model.named_steps["features"]
    tfidf = preprocessor.transformers_[0][1]
    tfidf_names = tfidf.get_feature_names_out().tolist()

    # Get source encoder feature names
    try:
        source_enc = preprocessor.transformers_[1][1]
        source_names = source_enc.get_feature_names_out().tolist()
    except (IndexError, AttributeError):
        source_names = []

    all_names = tfidf_names + source_names
    classifier = model.named_steps["classifier"]
    class_names = classifier.classes_.tolist()
    coef_matrix = classifier.coef_  # shape: (n_classes, n_features)

    result = {}
    for i, cls in enumerate(class_names):
        coefs = coef_matrix[i]
        # Only include features up to the number of names we have
        pairs = list(zip(all_names[:len(coefs)], coefs.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        result[cls] = pairs

    return result
