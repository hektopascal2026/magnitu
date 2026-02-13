"""
Recipe Distiller: converts a trained scikit-learn model into a lightweight
keyword-weight JSON recipe that seismo's PHP can evaluate.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import db
from config import MODELS_DIR, get_config
from pipeline import load_active_model, get_feature_importance, score_entries


def distill_recipe(top_n: int = None):
    """
    Extract top keywords per class from the active model
    and package them as a JSON recipe for seismo.

    Returns the recipe dict, or None if no active model.
    """
    config = get_config()
    if top_n is None:
        top_n = config.get("recipe_top_keywords", 200)

    model = load_active_model()
    if model is None:
        return None

    model_info = db.get_active_model()
    if not model_info:
        return None

    # Get feature importance per class
    importance = get_feature_importance()
    if not importance:
        return None

    class_names = list(importance.keys())

    # Build keyword map: {keyword: {class: weight, ...}}
    keywords = {}
    source_weights = {}

    for cls, pairs in importance.items():
        for feature, weight in pairs[:top_n]:
            if abs(weight) < 0.01:
                continue

            # Separate source-type features from text features
            if feature.startswith("source_type_") or feature.startswith("x0_"):
                # One-hot encoded source feature
                source_name = feature.replace("source_type_", "").replace("x0_", "")
                if source_name not in source_weights:
                    source_weights[source_name] = {}
                source_weights[source_name][cls] = round(float(weight), 4)
            else:
                if feature not in keywords:
                    keywords[feature] = {}
                keywords[feature][cls] = round(float(weight), 4)

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
    recipe_filename = f"recipe_v{model_info['version']}.json"
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


def evaluate_recipe_quality(recipe: dict, sample_size: int = 100) -> float:
    """
    Compare recipe-based scoring against full model scoring on a sample.
    Returns correlation score (0-1) indicating how well recipe approximates the model.
    """
    model = load_active_model()
    if model is None:
        return 0.0

    entries = db.get_all_entries()[:sample_size]
    if not entries:
        return 0.0

    # Get full model scores
    full_scores = score_entries(entries)
    if not full_scores:
        return 0.0

    # Get recipe-based scores (simplified scoring matching seismo's PHP logic)
    recipe_scores = []
    keywords = recipe.get("keywords", {})
    source_weights_map = recipe.get("source_weights", {})
    classes = recipe.get("classes", ["investigation_lead", "important", "background", "noise"])
    class_wts = recipe.get("class_weights", [1.0, 0.66, 0.33, 0.0])

    for entry in entries:
        text = f"{entry.get('title', '')} {entry.get('description', '')} {entry.get('content', '')}".lower()
        tokens = text.split()
        # Add bigrams
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        all_tokens = tokens + bigrams

        class_scores = {c: 0.0 for c in classes}
        for token in all_tokens:
            if token in keywords:
                for cls, wt in keywords[token].items():
                    if cls in class_scores:
                        class_scores[cls] += wt

        src = entry.get("source_type", "")
        if src in source_weights_map:
            for cls, wt in source_weights_map[src].items():
                if cls in class_scores:
                    class_scores[cls] += wt

        # Softmax
        max_s = max(class_scores.values()) if class_scores else 0
        exp_scores = {c: np.exp(s - max_s) for c, s in class_scores.items()}
        exp_sum = sum(exp_scores.values())
        probs = {c: exp_scores[c] / exp_sum if exp_sum > 0 else 1/len(classes) for c in classes}

        composite = sum(probs.get(c, 0) * class_wts[i] for i, c in enumerate(classes))
        recipe_scores.append(composite)

    # Compute correlation between full model and recipe scores
    full_vals = [s["relevance_score"] for s in full_scores[:len(recipe_scores)]]

    if len(full_vals) < 2:
        return 0.0

    correlation = np.corrcoef(full_vals, recipe_scores[:len(full_vals)])[0, 1]
    quality = max(0.0, float(correlation)) if not np.isnan(correlation) else 0.0

    return round(quality, 4)
