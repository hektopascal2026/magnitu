"""
Explainability module: per-entry explanations and global keyword rankings.
"""
import numpy as np
from pipeline import load_active_model, _prepare_text, get_feature_importance


def explain_entry(entry: dict):
    """
    Explain why a single entry received its score.
    Returns top contributing features with weights and directions.
    """
    model = load_active_model()
    if model is None:
        return None

    df = _prepare_text([entry])
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    class_names = model.classes_.tolist()

    # Get the feature matrix for this entry
    preprocessor = model.named_steps["features"]
    features = preprocessor.transform(df)

    # Get feature names
    tfidf = preprocessor.transformers_[0][1]
    tfidf_names = tfidf.get_feature_names_out().tolist()
    try:
        source_enc = preprocessor.transformers_[1][1]
        source_names = source_enc.get_feature_names_out().tolist()
    except (IndexError, AttributeError):
        source_names = []
    all_names = tfidf_names + source_names

    # Get coefficients for the predicted class
    classifier = model.named_steps["classifier"]
    pred_idx = class_names.index(prediction)
    coefs = classifier.coef_[pred_idx]

    # Compute feature contributions: coefficient * feature_value
    if hasattr(features, "toarray"):
        feature_values = features.toarray()[0]
    else:
        feature_values = np.array(features[0]).flatten()

    contributions = coefs[:len(feature_values)] * feature_values[:len(coefs)]

    # Get top contributing features
    n_features = min(len(all_names), len(contributions))
    feature_contribs = []
    for j in range(n_features):
        if abs(contributions[j]) > 0.001:
            feature_contribs.append({
                "feature": all_names[j],
                "weight": round(float(contributions[j]), 4),
                "direction": "positive" if contributions[j] > 0 else "negative",
            })

    # Sort by absolute contribution
    feature_contribs.sort(key=lambda x: abs(x["weight"]), reverse=True)
    top_features = feature_contribs[:8]

    probs = dict(zip(class_names, [round(float(p), 4) for p in probabilities]))

    # Composite relevance score
    class_weights = {"investigation_lead": 1.0, "important": 0.66, "background": 0.33, "noise": 0.0}
    relevance = sum(probs.get(c, 0) * class_weights.get(c, 0) for c in class_names)

    return {
        "prediction": prediction,
        "confidence": round(float(max(probabilities)), 4),
        "relevance_score": round(relevance, 4),
        "probabilities": probs,
        "top_features": top_features,
    }


def global_keywords(class_name: str = None, limit: int = 50) -> dict:
    """
    Get top keywords across all classes or for a specific class.
    Returns {class: [(keyword, weight), ...]} limited to top N per class.
    """
    importance = get_feature_importance()
    if not importance:
        return {}

    if class_name and class_name in importance:
        return {class_name: importance[class_name][:limit]}

    return {cls: pairs[:limit] for cls, pairs in importance.items()}


def compare_models(version_a: int, version_b: int) -> dict:
    """
    Compare keywords between two model versions.
    Returns changes: new keywords, removed keywords, weight shifts.
    """
    import db
    import joblib

    model_a = db.get_all_models()
    model_b = db.get_all_models()

    path_a = None
    path_b = None
    for m in model_a:
        if m["version"] == version_a:
            path_a = m["model_path"]
        if m["version"] == version_b:
            path_b = m["model_path"]

    if not path_a or not path_b:
        return {"error": "Model version not found"}

    # Load both models and compare top keywords
    # Simplified: just compare top 30 keywords per class
    from pathlib import Path
    if not Path(path_a).exists() or not Path(path_b).exists():
        return {"error": "Model file not found on disk"}

    pipe_a = joblib.load(path_a)
    pipe_b = joblib.load(path_b)

    def _top_keywords(pipe, n=30):
        tfidf = pipe.named_steps["features"].transformers_[0][1]
        names = tfidf.get_feature_names_out().tolist()
        clf = pipe.named_steps["classifier"]
        result = {}
        for i, cls in enumerate(clf.classes_):
            coefs = clf.coef_[i]
            pairs = sorted(zip(names[:len(coefs)], coefs), key=lambda x: abs(x[1]), reverse=True)
            result[cls] = dict(pairs[:n])
        return result

    kw_a = _top_keywords(pipe_a)
    kw_b = _top_keywords(pipe_b)

    changes = {}
    all_classes = set(list(kw_a.keys()) + list(kw_b.keys()))
    for cls in all_classes:
        a = kw_a.get(cls, {})
        b = kw_b.get(cls, {})
        new_kw = {k: v for k, v in b.items() if k not in a}
        removed_kw = {k: v for k, v in a.items() if k not in b}
        shifted = {}
        for k in set(a.keys()) & set(b.keys()):
            diff = b[k] - a[k]
            if abs(diff) > 0.05:
                shifted[k] = {"old": round(a[k], 4), "new": round(b[k], 4), "change": round(diff, 4)}
        changes[cls] = {"new": new_kw, "removed": removed_kw, "shifted": shifted}

    return {
        "version_a": version_a,
        "version_b": version_b,
        "changes": changes,
    }
