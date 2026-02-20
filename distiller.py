"""
Recipe Distiller: converts a trained model into a lightweight keyword-weight
JSON recipe that seismo's PHP can evaluate.

Magnitu 2: when the active model is a transformer, uses knowledge distillation.
A TF-IDF 'student' model is trained on the transformer's predictions across all
entries.  The recipe is then extracted from the student's coefficients — same
format, same PHP, but informed by transformer-quality classifications.

Also incorporates user reasoning text: key phrases from reasoning annotations
are boosted in the recipe so human-stated priorities carry extra weight.
"""
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import db
from config import MODELS_DIR, get_config
from pipeline import (
    load_active_model,
    get_feature_importance,
    score_entries,
    train_tfidf_student,
    build_tfidf_pipeline,
    _prepare_text,
)

logger = logging.getLogger(__name__)


def distill_recipe(top_n: Optional[int] = None):
    """
    Extract top keywords per class from the active model and package them as
    a JSON recipe for seismo.

    For TF-IDF models: extracts directly from coefficients (Magnitu 1 path).
    For transformer models: trains a TF-IDF student via knowledge distillation,
    then extracts from the student's coefficients (Magnitu 2 path).

    Returns the recipe dict, or None if no active model.
    """
    config = get_config()
    if top_n is None:
        top_n = config.get("recipe_top_keywords", 200)

    model_info = db.get_active_model()
    if not model_info:
        return None

    arch = model_info.get("architecture", "tfidf")

    if arch == "transformer":
        importance = _distill_from_transformer(top_n)
    else:
        importance = get_feature_importance()

    if not importance:
        return None

    # Build keyword map: {keyword: {class: weight, ...}}
    keywords = {}
    source_weights = {}

    for cls, pairs in importance.items():
        for feature, weight in pairs[:top_n]:
            if abs(weight) < 0.01:
                continue

            if feature.startswith("source_type_") or feature.startswith("x0_"):
                source_name = feature.replace("source_type_", "").replace("x0_", "")
                if source_name not in source_weights:
                    source_weights[source_name] = {}
                source_weights[source_name][cls] = round(float(weight), 4)
            else:
                if feature not in keywords:
                    keywords[feature] = {}
                keywords[feature][cls] = round(float(weight), 4)

    # Boost keywords from user reasoning annotations
    keywords = _boost_from_reasoning(keywords)

    # Normalize weights so accumulated scores stay in a reasonable range for
    # softmax regardless of text length — prevents long entries from getting
    # extreme scores while short entries cluster near 0.5
    keywords, source_weights = _normalize_weights(keywords, source_weights)

    # Build recipe
    recipe = {
        "version": model_info["version"],
        "trained_at": model_info.get("trained_at", datetime.now().isoformat()),
        "labels_used": model_info.get("label_count", 0),
        "metrics": {
            "accuracy": round(model_info.get("accuracy", 0), 4),
            "f1_macro": round(model_info.get("f1_score", 0), 4),
        },
        "alert_threshold": config.get("alert_threshold", 0.75),
        "classes": ["investigation_lead", "important", "background", "noise"],
        "class_weights": [1.0, 0.66, 0.33, 0.0],
        "keywords": keywords,
        "source_weights": source_weights,
    }

    # Save recipe to disk
    recipe_filename = "recipe_v{}.json".format(model_info["version"])
    recipe_path = str(MODELS_DIR / recipe_filename)
    with open(recipe_path, "w") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)

    # Update model record with recipe path
    conn = db.get_db()
    conn.execute(
        "UPDATE models SET recipe_path = ? WHERE version = ?",
        (recipe_path, model_info["version"])
    )
    conn.commit()
    conn.close()

    return recipe


def _normalize_weights(keywords: dict, source_weights: dict) -> tuple:
    """
    Scale recipe weights so that the median accumulated class_score across all
    entries lands near 1.0 before softmax.  This prevents long entries (many
    keyword matches) from producing extremely peaked softmax distributions
    while short entries get near-uniform scores.

    Without this, a 500-token government press release can score 100 while
    a 50-token news teaser about the same topic scores 5.
    """
    all_entries = db.get_all_entries()
    if not all_entries or not keywords:
        return keywords, source_weights

    classes = ["investigation_lead", "important", "background", "noise"]
    magnitudes = []

    for entry in all_entries:
        text = "{} {} {}".format(
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("content", ""),
        ).lower()
        tokens = text.split()
        bigrams = ["{} {}".format(tokens[j], tokens[j + 1])
                   for j in range(len(tokens) - 1)]
        all_tokens = tokens + bigrams

        class_scores = {c: 0.0 for c in classes}
        for token in all_tokens:
            if token in keywords:
                for cls, wt in keywords[token].items():
                    if cls in class_scores:
                        class_scores[cls] += wt

        src = entry.get("source_type", "")
        if src in source_weights:
            for cls, wt in source_weights[src].items():
                if cls in class_scores:
                    class_scores[cls] += wt

        mag = max(abs(v) for v in class_scores.values()) if class_scores else 0
        if mag > 0:
            magnitudes.append(mag)

    if not magnitudes:
        return keywords, source_weights

    magnitudes.sort()
    median_mag = magnitudes[len(magnitudes) // 2]

    TARGET_MAGNITUDE = 2.0
    if median_mag < 0.01:
        return keywords, source_weights

    scale = TARGET_MAGNITUDE / median_mag

    logger.info(
        "Recipe normalization: median magnitude %.2f, scaling weights by %.3f",
        median_mag, scale,
    )

    normalized_kw = {}
    for word, class_weights in keywords.items():
        normalized_kw[word] = {
            cls: round(wt * scale, 4)
            for cls, wt in class_weights.items()
        }

    normalized_sw = {}
    for src, class_weights in source_weights.items():
        normalized_sw[src] = {
            cls: round(wt * scale, 4)
            for cls, wt in class_weights.items()
        }

    return normalized_kw, normalized_sw


def _distill_from_transformer(top_n: int) -> dict:
    """
    Knowledge distillation: train a TF-IDF student from the transformer's
    predictions, then extract feature importance from the student.
    """
    logger.info("Knowledge distillation: training TF-IDF student from transformer...")
    student = train_tfidf_student()
    if student is None:
        logger.warning("Could not train TF-IDF student for distillation.")
        return {}

    # Extract feature importance from the student
    preprocessor = student.named_steps["features"]
    tfidf = preprocessor.transformers_[0][1]
    tfidf_names = tfidf.get_feature_names_out().tolist()

    try:
        source_enc = preprocessor.transformers_[1][1]
        source_names = source_enc.get_feature_names_out().tolist()
    except (IndexError, AttributeError):
        source_names = []

    all_names = tfidf_names + source_names
    classifier = student.named_steps["classifier"]
    class_names = classifier.classes_.tolist()
    coef_matrix = classifier.coef_

    result = {}
    for i, cls in enumerate(class_names):
        coefs = coef_matrix[i]
        pairs = list(zip(all_names[:len(coefs)], coefs.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        result[cls] = pairs

    logger.info("Knowledge distillation complete. %d features extracted.", len(all_names))
    return result


def _boost_from_reasoning(keywords: dict) -> dict:
    """
    Extract key phrases from user reasoning annotations and boost their
    weights in the recipe.  This ensures that explicitly-stated reasons
    ('links politician X to company Y') increase the recipe's sensitivity
    to those terms.
    """
    reasoning_labels = db.get_all_reasoning_texts()
    if not reasoning_labels:
        return keywords

    BOOST_FACTOR = 1.5

    for rl in reasoning_labels:
        reasoning = rl.get("reasoning", "")
        label = rl.get("label", "")
        if not reasoning or not label:
            continue

        # Tokenize reasoning into words (simple split, lowercase)
        tokens = re.findall(r'\b[a-zA-Z\u00C0-\u024F]{3,}\b', reasoning.lower())

        for token in tokens:
            if token in keywords:
                # Boost existing keyword's weight for this class
                if label in keywords[token]:
                    keywords[token][label] = round(
                        keywords[token][label] * BOOST_FACTOR, 4
                    )
            else:
                # Add as new keyword with a moderate base weight
                keywords[token] = {label: 0.1}

    return keywords


def evaluate_recipe_quality(recipe: dict, sample_size: int = 100) -> float:
    """
    Compare recipe-based scoring against full model scoring on a sample.
    Returns correlation score (0-1) indicating how well recipe approximates the model.
    """
    entries = db.get_all_entries()[:sample_size]
    if not entries:
        return 0.0

    full_scores = score_entries(entries)
    if not full_scores:
        return 0.0

    # Build lookup of model scores keyed by (entry_type, entry_id)
    model_score_map = {
        (s["entry_type"], s["entry_id"]): s["relevance_score"]
        for s in full_scores
    }

    # Compute recipe-based scores (matching Seismo's PHP logic)
    kw = recipe.get("keywords", {})
    source_weights_map = recipe.get("source_weights", {})
    classes = recipe.get("classes", ["investigation_lead", "important", "background", "noise"])
    class_wts = recipe.get("class_weights", [1.0, 0.66, 0.33, 0.0])

    paired_model = []
    paired_recipe = []

    for entry in entries:
        key = (entry["entry_type"], entry["entry_id"])
        if key not in model_score_map:
            continue

        text = "{} {} {}".format(
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("content", ""),
        ).lower()
        tokens = text.split()
        bigrams = ["{} {}".format(tokens[j], tokens[j + 1]) for j in range(len(tokens) - 1)]
        all_tokens = tokens + bigrams

        class_scores = {c: 0.0 for c in classes}
        for token in all_tokens:
            if token in kw:
                for cls, wt in kw[token].items():
                    if cls in class_scores:
                        class_scores[cls] += wt

        src = entry.get("source_type", "")
        if src in source_weights_map:
            for cls, wt in source_weights_map[src].items():
                if cls in class_scores:
                    class_scores[cls] += wt

        max_s = max(class_scores.values()) if class_scores else 0
        exp_scores = {c: np.exp(s - max_s) for c, s in class_scores.items()}
        exp_sum = sum(exp_scores.values())
        probs = {
            c: exp_scores[c] / exp_sum if exp_sum > 0 else 1 / len(classes)
            for c in classes
        }

        composite = sum(probs.get(c, 0) * class_wts[idx] for idx, c in enumerate(classes))
        paired_model.append(model_score_map[key])
        paired_recipe.append(composite)

    if len(paired_model) < 2:
        return 0.0

    correlation = float(np.corrcoef(paired_model, paired_recipe)[0, 1])
    quality = max(0.0, correlation) if not (correlation != correlation) else 0.0

    return round(quality, 4)
