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
async def top_page(request: Request, view: str = "recent"):
    """Top entries with multiple views for model validation and improvement."""
    ctx = _base_context(request)

    # Validate view parameter
    if view not in ("recent", "mismatches", "all"):
        view = "recent"
    ctx["view"] = view

    if view == "recent":
        # Recent unlabeled top 30: highest-scored entries from last 7 days, unlabeled only
        entries = db.get_recent_entries(days=7)
        scored = pipeline.score_entries(entries)
        # Build a lookup for quick label checks (batch query)
        all_labels = {(l["entry_type"], l["entry_id"]) for l in db.get_all_labels_raw()}
        labeled_ids = set()
        for e in entries:
            if (e["entry_type"], e["entry_id"]) in all_labels:
                labeled_ids.add((e["entry_type"], e["entry_id"]))
        # Filter to unlabeled, sort by score
        unlabeled_scored = [s for s in scored
                           if (s["entry_type"], s["entry_id"]) not in labeled_ids]
        unlabeled_scored.sort(key=lambda s: s["relevance_score"], reverse=True)
        top_scored = unlabeled_scored[:30]
        # Build entry map for fast lookup
        entry_map = {(e["entry_type"], e["entry_id"]): e for e in entries}
        top_entries = []
        for s in top_scored:
            entry = entry_map.get((s["entry_type"], s["entry_id"]))
            if not entry:
                continue
            top_entries.append({
                "entry": entry,
                "score": s,
                "user_label": None,
                "match": None,
            })
        ctx["top_entries"] = top_entries
        ctx["labeled_count"] = 0
        ctx["correct_count"] = 0
        ctx["accuracy"] = None
        ctx["total_recent"] = len(entries)
        ctx["total_recent_unlabeled"] = len(unlabeled_scored)

    elif view == "mismatches":
        # Mismatches: entries where user label != model prediction
        labeled_entries = db.get_labeled_entries()
        scored = pipeline.score_entries(labeled_entries)
        # Build score lookup
        score_map = {(s["entry_type"], s["entry_id"]): s for s in scored}
        top_entries = []
        for entry in labeled_entries:
            key = (entry["entry_type"], entry["entry_id"])
            s = score_map.get(key)
            if not s:
                continue
            user_label = entry["user_label"]
            predicted = s["predicted_label"]
            if user_label != predicted:
                top_entries.append({
                    "entry": entry,
                    "score": s,
                    "user_label": user_label,
                    "match": False,
                })
        # Sort by score descending (high-impact mismatches first)
        top_entries.sort(key=lambda x: x["score"]["relevance_score"], reverse=True)
        total_mismatches = len(top_entries)
        top_entries = top_entries[:30]
        ctx["top_entries"] = top_entries
        ctx["labeled_count"] = len(top_entries)
        ctx["correct_count"] = 0
        ctx["accuracy"] = None
        ctx["total_mismatches"] = total_mismatches

    else:
        # All-time top 30 (original behavior)
        all_entries = db.get_all_entries()
        scored = pipeline.score_entries(all_entries)
        scored.sort(key=lambda s: s["relevance_score"], reverse=True)
        top_scored = scored[:30]
        entry_map = {(e["entry_type"], e["entry_id"]): e for e in all_entries}
        top_entries = []
        for s in top_scored:
            entry = entry_map.get((s["entry_type"], s["entry_id"]))
            if not entry:
                continue
            user_label = db.get_label(s["entry_type"], s["entry_id"])
            top_entries.append({
                "entry": entry,
                "score": s,
                "user_label": user_label,
                "match": user_label == s["predicted_label"] if user_label else None,
            })
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
        labels_synced = 0
        try:
            labels_synced = await loop.run_in_executor(None, sync.pull_labels)
        except Exception:
            pass  # Seismo might not have the labels endpoint yet

        # Check recent sync log for conflict details
        syncs = db.get_recent_syncs(1)
        sync_detail = syncs[0]["details"] if syncs else ""

        return {
            "success": True,
            "entries_fetched": count,
            "labels_synced": labels_synced,
            "sync_detail": sync_detail,
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

    # Build model metadata for Seismo — includes quality metrics so Seismo
    # can gate whether to accept scores from a weaker model
    profile = model_manager.get_profile()
    model_meta = None
    if profile:
        model_meta = {
            "model_name": profile.get("model_name", ""),
            "model_uuid": profile.get("model_uuid", ""),
            "model_description": profile.get("description", ""),
            "model_version": model_info["version"],
            "model_trained_at": model_info.get("trained_at", ""),
            "accuracy": model_info.get("accuracy", 0.0),
            "f1_score": model_info.get("f1_score", 0.0),
            "label_count": model_info.get("label_count", 0),
            "architecture": model_info.get("architecture", "tfidf"),
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
    else:
        arch = result.get("architecture", "tfidf")
        if arch == "transformer":
            result["recipe_note"] = (
                "Recipe not generated yet. Knowledge distillation needs at least "
                "20 entries to build a recipe. Sync more entries from Seismo, then retrain."
            )
        else:
            result["recipe_note"] = "No recipe generated — model may not have enough data."

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
    old_transformer_name = config.get("transformer_model_name", "")

    for key in ["seismo_url", "api_key", "min_labels_to_train",
                "recipe_top_keywords", "auto_train_after_n_labels", "alert_threshold",
                "model_architecture", "transformer_model_name"]:
        if key in data:
            config[key] = data[key]
    save_config(config)

    # If the transformer model name changed, invalidate all cached embeddings
    # and clear the in-memory model so the new one gets loaded on next use
    new_transformer_name = config.get("transformer_model_name", "")
    if new_transformer_name != old_transformer_name and old_transformer_name:
        db.invalidate_all_embeddings()
        pipeline.invalidate_embedder_cache()

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

def _migrate_config():
    """One-time config migrations that run on startup."""
    cfg = get_config()
    changed = False

    # v2.1: switch from English-only distilroberta-base to multilingual
    # xlm-roberta-base so German/French entries get proper embeddings.
    if cfg.get("transformer_model_name") == "distilroberta-base":
        import logging
        logging.getLogger(__name__).info(
            "Migrating transformer model: distilroberta-base → xlm-roberta-base"
        )
        cfg["transformer_model_name"] = "xlm-roberta-base"
        changed = True

    if changed:
        save_config(cfg)
        db.invalidate_all_embeddings()
        pipeline.invalidate_embedder_cache()


@app.on_event("startup")
async def startup():
    db.init_db()
    _migrate_config()
