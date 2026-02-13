"""
Magnitu — ML-powered relevance scoring for Seismo.
FastAPI application: serves the labeling UI, dashboard, and orchestrates ML pipeline.
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json

import db
import sync
import pipeline
import explainer
import distiller
from config import get_config, save_config, BASE_DIR

app = FastAPI(title="Magnitu", version="0.1.0")

# Static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ─── Template helpers ───

def _base_context(request: Request) -> dict:
    """Common context for all templates."""
    config = get_config()
    active_model = db.get_active_model()
    return {
        "request": request,
        "config": config,
        "label_count": db.get_label_count(),
        "entry_count": db.get_entry_count(),
        "active_model": active_model,
        "label_distribution": db.get_label_distribution(),
    }


# ─── Pages ───

@app.get("/", response_class=HTMLResponse)
async def labeling_page(request: Request):
    """Main labeling page."""
    ctx = _base_context(request)
    entries = db.get_unlabeled_entries(limit=30)

    # Add existing labels and explanations if model exists
    for entry in entries:
        entry["_label"] = db.get_label(entry["entry_type"], entry["entry_id"])

    ctx["entries"] = entries
    ctx["unlabeled_count"] = len([e for e in entries if e["_label"] is None])
    ctx["today_labels"] = _today_label_count()
    return templates.TemplateResponse("labeling.html", ctx)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Model stats, keywords, and training history."""
    ctx = _base_context(request)
    ctx["models"] = db.get_all_models()
    ctx["syncs"] = db.get_recent_syncs(20)

    # Get keyword data if model exists
    ctx["keywords"] = {}
    if ctx["active_model"]:
        try:
            kw = explainer.global_keywords(limit=30)
            ctx["keywords"] = kw
        except Exception:
            pass

    return templates.TemplateResponse("dashboard.html", ctx)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings: seismo connection, thresholds."""
    ctx = _base_context(request)
    ctx["syncs"] = db.get_recent_syncs(10)
    return templates.TemplateResponse("settings.html", ctx)


# ─── API: Labeling ───

@app.post("/api/label")
async def set_label(
    entry_type: str = Form(...),
    entry_id: int = Form(...),
    label: str = Form(...),
):
    """Set or update a label for an entry."""
    valid_labels = ["investigation_lead", "important", "background", "noise"]
    if label not in valid_labels:
        raise HTTPException(400, f"Invalid label. Must be one of: {valid_labels}")
    db.set_label(entry_type, entry_id, label)
    return {"success": True, "entry_type": entry_type, "entry_id": entry_id, "label": label}


@app.post("/api/unlabel")
async def remove_label(
    entry_type: str = Form(...),
    entry_id: int = Form(...),
):
    """Remove a label from an entry."""
    db.remove_label(entry_type, entry_id)
    return {"success": True}


# ─── API: Sync ───

@app.post("/api/sync/pull")
async def sync_pull():
    """Pull entries from Seismo."""
    try:
        count = sync.pull_entries()
        return {"success": True, "entries_fetched": count}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/sync/push")
async def sync_push():
    """Score all entries and push scores + recipe to Seismo."""
    model_info = db.get_active_model()
    if not model_info:
        raise HTTPException(400, "No trained model. Train first.")

    # Score all entries
    all_entries = db.get_all_entries()
    if not all_entries:
        raise HTTPException(400, "No entries to score.")

    scores = pipeline.score_entries(all_entries)

    # Add explanations
    for i, entry in enumerate(all_entries):
        exp = explainer.explain_entry(entry)
        if exp and i < len(scores):
            scores[i]["explanation"] = {
                "top_features": exp["top_features"],
                "confidence": exp["confidence"],
                "prediction": exp["prediction"],
            }

    # Push scores
    try:
        score_result = sync.push_scores(scores, model_info["version"])
    except Exception as e:
        raise HTTPException(500, f"Failed to push scores: {e}")

    # Distill and push recipe
    recipe = distiller.distill_recipe()
    recipe_result = {}
    if recipe:
        try:
            recipe_result = sync.push_recipe(recipe)
        except Exception as e:
            recipe_result = {"error": str(e)}

    return {
        "success": True,
        "scores_pushed": len(scores),
        "score_result": score_result,
        "recipe_result": recipe_result,
    }


@app.get("/api/sync/test")
async def sync_test():
    """Test connection to Seismo."""
    ok, msg = sync.test_connection()
    return {"success": ok, "message": msg}


# ─── API: Training ───

@app.post("/api/train")
async def train_model():
    """Train a new model on all labeled entries."""
    result = pipeline.train()
    if not result.get("success"):
        return JSONResponse(result, status_code=400)

    # Auto-distill recipe after training
    recipe = distiller.distill_recipe()
    if recipe:
        quality = distiller.evaluate_recipe_quality(recipe)
        result["recipe_version"] = recipe.get("version")
        result["recipe_quality"] = quality
        result["recipe_keywords"] = len(recipe.get("keywords", {}))

    return result


@app.get("/api/explain/{entry_type}/{entry_id}")
async def explain(entry_type: str, entry_id: int):
    """Get explanation for a specific entry."""
    # Find entry in local DB
    conn = db.get_db()
    row = conn.execute(
        "SELECT * FROM entries WHERE entry_type = ? AND entry_id = ?",
        (entry_type, entry_id)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Entry not found")

    entry = dict(row)
    exp = explainer.explain_entry(entry)
    if not exp:
        raise HTTPException(400, "No active model")
    return exp


@app.get("/api/keywords")
async def keywords(class_name: str = None, limit: int = 50):
    """Get top keywords, optionally filtered by class."""
    kw = explainer.global_keywords(class_name, limit)
    return kw


@app.get("/api/stats")
async def stats():
    """Get current system stats."""
    model = db.get_active_model()
    return {
        "entries": db.get_entry_count(),
        "labels": db.get_label_count(),
        "label_distribution": db.get_label_distribution(),
        "active_model": model,
        "models_count": len(db.get_all_models()),
    }


# ─── API: Settings ───

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update configuration."""
    data = await request.json()
    config = get_config()
    for key in ["seismo_url", "api_key", "min_labels_to_train",
                "recipe_top_keywords", "auto_train_after_n_labels", "alert_threshold"]:
        if key in data:
            config[key] = data[key]
    save_config(config)
    return {"success": True, "config": config}


# ─── Helpers ───

def _today_label_count() -> int:
    """Count labels created today."""
    from datetime import date
    today = date.today().isoformat()
    conn = db.get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM labels WHERE date(updated_at) = ?", (today,)
    ).fetchone()[0]
    conn.close()
    return count


# ─── Startup ───

@app.on_event("startup")
async def startup():
    db.init_db()
