"""
Magnitu 2 — ML-powered relevance scoring for Seismo.
FastAPI application: serves the labeling UI, dashboard, and orchestrates ML pipeline.
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
from typing import Optional

import db
import sync
import pipeline
import explainer
import distiller
import sampler
import model_manager
from config import get_config, save_config, BASE_DIR, VERSION

app = FastAPI(title="Magnitu", version=VERSION)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ─── Template helpers ───

def _base_context(request: Request) -> dict:
    """Common context for all templates."""
    config = get_config()
    active_model = db.get_active_model()
    profile = model_manager.get_profile()
    return {
        "request": request,
        "config": config,
        "version": VERSION,
        "label_count": db.get_label_count(),
        "entry_count": db.get_entry_count(),
        "active_model": active_model,
        "label_distribution": db.get_label_distribution(),
        "profile": profile,
        "architecture": config.get("model_architecture", "transformer"),
        "embedding_count": db.get_embedding_count(),
    }


# ─── Pages ───

@app.get("/", response_class=HTMLResponse)
async def labeling_page(request: Request):
    """Main labeling page — smart-sampled for active learning."""
    # First-run: redirect to setup if no model profile exists
    if not model_manager.has_profile():
        return RedirectResponse("/setup", status_code=302)
    ctx = _base_context(request)
    entries = sampler.get_smart_entries(limit=30)

    # Add existing labels and reasoning
    for entry in entries:
        label_data = db.get_label_with_reasoning(entry["entry_type"], entry["entry_id"])
        entry["_label"] = label_data["label"] if label_data else None
        entry["_reasoning"] = label_data["reasoning"] if label_data else ""

    ctx["entries"] = entries
    ctx["unlabeled_count"] = len([e for e in entries if e["_label"] is None])
    ctx["today_labels"] = _today_label_count()

    # Sampling stats for the UI
    reasons = {}
    for e in entries:
        r = e.get("_sampling_reason", "new")
        reasons[r] = reasons.get(r, 0) + 1
    ctx["sampling_stats"] = reasons
    ctx["has_model"] = ctx["active_model"] is not None

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


@app.get("/top", response_class=HTMLResponse)
async def top_page(request: Request):
    """Top 30 highest-scored entries — validates model accuracy."""
    ctx = _base_context(request)
    
    # Score all local entries with the active model
    all_entries = db.get_all_entries()
    scored = pipeline.score_entries(all_entries)
    
    # Sort by relevance score descending, take top 30
    scored.sort(key=lambda s: s["relevance_score"], reverse=True)
    top_scored = scored[:30]
    
    # Enrich with entry data and user labels
    top_entries = []
    for s in top_scored:
        # Find the full entry
        entry = next((e for e in all_entries 
                       if e["entry_type"] == s["entry_type"] and e["entry_id"] == s["entry_id"]), None)
        if not entry:
            continue
        user_label = db.get_label(s["entry_type"], s["entry_id"])
        top_entries.append({
            "entry": entry,
            "score": s,
            "user_label": user_label,
            "match": user_label == s["predicted_label"] if user_label else None,
        })
    
    # Accuracy on labeled subset
    labeled_in_top = [e for e in top_entries if e["user_label"] is not None]
    correct = sum(1 for e in labeled_in_top if e["match"])
    ctx["top_entries"] = top_entries
    ctx["labeled_count"] = len(labeled_in_top)
    ctx["correct_count"] = correct
    ctx["accuracy"] = round(correct / len(labeled_in_top) * 100, 1) if labeled_in_top else None
    
    return templates.TemplateResponse("top.html", ctx)


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """How Magnitu learns — explains the ML pipeline to the user."""
    ctx = _base_context(request)
    return templates.TemplateResponse("about.html", ctx)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings: seismo connection, thresholds."""
    ctx = _base_context(request)
    ctx["syncs"] = db.get_recent_syncs(10)
    return templates.TemplateResponse("settings.html", ctx)


@app.get("/model", response_class=HTMLResponse)
async def model_page(request: Request):
    """Model profile, export/import, version history."""
    if not model_manager.has_profile():
        return RedirectResponse("/setup", status_code=302)
    ctx = _base_context(request)
    ctx["models"] = db.get_all_models()
    return templates.TemplateResponse("model.html", ctx)


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    """First-run setup: create or load a model."""
    # If profile already exists, redirect to model page
    if model_manager.has_profile():
        return RedirectResponse("/model", status_code=302)
    ctx = _base_context(request)
    return templates.TemplateResponse("setup.html", ctx)


# ─── API: Model ───

@app.post("/api/model/create")
async def create_model(request: Request):
    """Create a new model profile."""
    data = await request.json()
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    if not name:
        return JSONResponse({"success": False, "error": "Model name is required."}, status_code=400)
    if model_manager.has_profile():
        return JSONResponse({"success": False, "error": "A model profile already exists."}, status_code=400)
    profile = model_manager.create_profile(name, description)
    return {"success": True, "profile": profile}


@app.post("/api/model/update")
async def update_model(request: Request):
    """Update model description. Name is immutable once set."""
    data = await request.json()
    description = data.get("description")
    model_manager.update_profile(description=description)
    return {"success": True}


@app.get("/api/model/export")
async def export_model():
    """Export the current model as a .magnitu file download."""
    from fastapi.responses import FileResponse
    try:
        path = model_manager.export_model()
        profile = model_manager.get_profile()
        safe_name = (profile["model_name"] if profile else "model").replace(" ", "_").lower()
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"{safe_name}.magnitu",
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/model/fork")
async def fork_model(request: Request):
    """Export the current model as a new model with a fresh identity."""
    from fastapi.responses import FileResponse
    data = await request.json()
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    if not name:
        return JSONResponse({"success": False, "error": "Model name is required."}, status_code=400)
    try:
        path = model_manager.export_as_new_model(name, description)
        safe_name = name.replace(" ", "_").lower()
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"{safe_name}.magnitu",
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/model/import")
async def import_model_upload(request: Request):
    """Import a .magnitu file (multipart upload)."""
    import asyncio
    import tempfile
    form = await request.form()
    upload = form.get("file")
    if not upload:
        return JSONResponse({"success": False, "error": "No file uploaded."}, status_code=400)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".magnitu") as tmp:
        content = await upload.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run blocking import in a thread so it doesn't freeze the server
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model_manager.import_model, tmp_path)
        return {"success": True, **result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─── API: Labeling ───

@app.post("/api/label")
async def set_label(
    entry_type: str = Form(...),
    entry_id: int = Form(...),
    label: str = Form(...),
    reasoning: str = Form(""),
):
    """Set or update a label for an entry, with optional reasoning."""
    valid_labels = ["investigation_lead", "important", "background", "noise"]
    if label not in valid_labels:
        raise HTTPException(400, "Invalid label. Must be one of: {}".format(valid_labels))
    db.set_label(entry_type, entry_id, label, reasoning=reasoning.strip())

    # Push label to Seismo in background (best-effort)
    try:
        sync.push_labels()
    except Exception:
        pass  # Don't fail the label action if push fails

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
    """Pull entries and labels from Seismo. In transformer mode, also computes embeddings."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, sync.pull_entries)
        # Also pull labels so new instances get existing training data
        labels_imported = 0
        try:
            labels_imported = await loop.run_in_executor(None, sync.pull_labels)
        except Exception:
            pass  # Seismo might not have the labels endpoint yet
        return {
            "success": True,
            "entries_fetched": count,
            "labels_imported": labels_imported,
            "embeddings": db.get_embedding_count(),
        }
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

    # Build model metadata for Seismo
    profile = model_manager.get_profile()
    model_meta = None
    if profile:
        model_meta = {
            "model_name": profile.get("model_name", ""),
            "model_description": profile.get("description", ""),
            "model_version": model_info["version"],
            "model_trained_at": model_info.get("trained_at", ""),
        }

    # Push scores
    try:
        score_result = sync.push_scores(scores, model_info["version"], model_meta=model_meta)
    except Exception as e:
        raise HTTPException(500, f"Failed to push scores: {e}")

    # Also push labels to keep Seismo in sync
    try:
        sync.push_labels()
    except Exception:
        pass

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


@app.post("/api/sync/labels")
async def sync_labels():
    """Push local labels to Seismo and pull any new labels back."""
    pushed = {}
    pulled = 0
    try:
        pushed = sync.push_labels()
    except Exception as e:
        raise HTTPException(500, f"Failed to push labels: {e}")
    try:
        pulled = sync.pull_labels()
    except Exception:
        pass
    return {"success": True, "pushed": pushed, "labels_imported": pulled}


@app.get("/api/sync/test")
async def sync_test():
    """Test connection to Seismo."""
    ok, msg = sync.test_connection()
    return {"success": ok, "message": msg}


# ─── API: Training ───

@app.post("/api/train")
async def train_model():
    """Train a new model on all labeled entries."""
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipeline.train)
    if not result.get("success"):
        return JSONResponse(result, status_code=400)

    # Auto-distill recipe after training
    recipe = await loop.run_in_executor(None, distiller.distill_recipe)
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
    config = get_config()
    model = db.get_active_model()
    return {
        "magnitu_version": VERSION,
        "architecture": config.get("model_architecture", "transformer"),
        "entries": db.get_entry_count(),
        "labels": db.get_label_count(),
        "embeddings": db.get_embedding_count(),
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
                "recipe_top_keywords", "auto_train_after_n_labels", "alert_threshold",
                "model_architecture", "transformer_model_name"]:
        if key in data:
            config[key] = data[key]
    save_config(config)
    return {"success": True, "config": config}


@app.post("/api/embeddings/compute")
async def compute_embeddings():
    """Compute embeddings for all entries that don't have them yet."""
    import asyncio
    config = get_config()
    if config.get("model_architecture") != "transformer":
        return {"success": False, "error": "Not in transformer mode."}

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sync._compute_pending_embeddings)
    return {
        "success": True,
        "embedding_count": db.get_embedding_count(),
        "entry_count": db.get_entry_count(),
    }


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
