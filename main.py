"""
Magnitu 2 — ML-powered relevance scoring for Seismo.
FastAPI application: serves the labeling UI, dashboard, and orchestrates ML pipeline.
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
from typing import Optional
import threading
import uuid
from datetime import datetime

import db
import sync
import pipeline
import explainer
import distiller
import sampler
import model_manager
from config import get_config, save_config, BASE_DIR, VERSION
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Magnitu", version=VERSION)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Background job registry for long-running UI actions (sync/push)
_JOB_LOCK = threading.Lock()
_JOBS = {}


def _create_job(job_type: str) -> str:
    job_id = uuid.uuid4().hex
    with _JOB_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "result": None,
            "error": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = datetime.utcnow().isoformat()


def _get_job(job_id: str) -> Optional[dict]:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _run_job(job_id: str, target):
    _update_job(job_id, status="running", progress=1, message="Starting...")
    try:
        def progress_cb(pct: int, msg: str):
            _update_job(job_id, progress=max(0, min(100, int(pct))), message=msg)

        result = target(progress_cb)
        _update_job(
            job_id,
            status="success",
            progress=100,
            message="Done",
            result=result,
            error=None,
        )
    except Exception as e:
        _update_job(
            job_id,
            status="error",
            message=str(e),
            error=str(e),
        )


def _sync_pull_impl(full: bool, progress_cb=None) -> dict:
    emb_before = db.get_embedding_count()
    count = 0
    entries_by_type = {}
    remote_total = 0
    embedding_rounds = 0

    if progress_cb:
        progress_cb(5, "Starting sync...")

    if full:
        # Full backfill mode: pull each source family with high limits, then
        # iterate embedding computation until no missing embeddings remain.
        try:
            status = sync.get_status()
            remote_entries = status.get("entries", {})
            remote_total = int(remote_entries.get("total", 0) or 0)
        except Exception:
            remote_entries = {}
            remote_total = 0

        type_specs = [
            ("feed_item", "feed_items"),
            ("email", "emails"),
            ("lex_item", "lex_items"),
        ]
        for idx, (entry_type, status_key) in enumerate(type_specs):
            expected = int(remote_entries.get(status_key, 0) or 0)
            limit = max(1000, expected + 250)
            if progress_cb:
                progress_cb(10 + idx * 20, "Pulling {} entries...".format(entry_type))
            fetched = sync.pull_entries(entry_type=entry_type, limit=limit)
            entries_by_type[entry_type] = fetched
            count += fetched

        if get_config().get("model_architecture") == "transformer":
            # Keep running embedding pass until no missing entries remain.
            while db.get_entries_without_embeddings(limit=1):
                if progress_cb:
                    missing_now = len(db.get_entries_without_embeddings(limit=5000))
                    progress_cb(
                        min(90, 70 + embedding_rounds * 4),
                        "Computing embeddings ({} missing)...".format(missing_now)
                    )
                sync._compute_pending_embeddings()
                embedding_rounds += 1
                if embedding_rounds >= 25:
                    break
    else:
        if progress_cb:
            progress_cb(25, "Pulling latest entries...")
        count = sync.pull_entries()
        embedding_rounds = 1

    if progress_cb:
        progress_cb(92, "Pulling labels...")
    labels_synced = 0
    try:
        labels_synced = sync.pull_labels()
    except Exception as e:
        logger.warning("Label pull failed during sync: %s", e)
        db.log_sync("pull", 0, "FAILED label pull: {}".format(e))

    syncs = db.get_recent_syncs(1)
    sync_detail = syncs[0]["details"] if syncs else ""

    emb_after = db.get_embedding_count()
    entry_count = db.get_entry_count()
    emb_computed = emb_after - emb_before
    emb_warning = ""
    if entry_count > 0 and emb_after < entry_count * 0.5:
        emb_warning = "Only {}/{} entries have embeddings. Model scoring will be limited.".format(
            emb_after, entry_count
        )

    result = {
        "success": True,
        "full_mode": full,
        "entries_fetched": count,
        "labels_synced": labels_synced,
        "sync_detail": sync_detail,
        "embeddings": emb_after,
        "embeddings_computed": emb_computed,
        "entry_count": entry_count,
        "embedding_warning": emb_warning,
        "embedding_rounds": embedding_rounds,
    }
    if full:
        result["entries_by_type"] = entries_by_type
        result["remote_total"] = remote_total
    return result


def _sync_push_impl(progress_cb=None) -> dict:
    import httpx as _httpx

    if progress_cb:
        progress_cb(5, "Preparing push...")

    model_info = db.get_active_model()
    if not model_info:
        raise ValueError("No trained model. Train first.")

    all_entries = db.get_all_entries()
    if not all_entries:
        raise ValueError("No entries to score.")

    cfg = get_config()
    if cfg.get("model_architecture") == "transformer":
        missing = db.get_entries_without_embeddings(limit=5000)
        if missing:
            if progress_cb:
                progress_cb(20, "Computing missing embeddings...")
            sync._compute_pending_embeddings()

    if progress_cb:
        progress_cb(45, "Scoring entries...")
    try:
        scores = pipeline.score_entries(all_entries)
    except Exception as e:
        raise ValueError("Scoring failed: {}".format(e))

    if not scores:
        emb_count = db.get_embedding_count()
        entry_count = db.get_entry_count()
        raise ValueError(
            "No scores produced. Embeddings: {}/{} entries. Try syncing again.".format(emb_count, entry_count)
        )

    if progress_cb:
        progress_cb(62, "Building explanations...")
    score_by_key = {(s["entry_type"], s["entry_id"]): s for s in scores}
    for entry in all_entries:
        key = (entry["entry_type"], entry["entry_id"])
        score = score_by_key.get(key)
        if not score:
            continue
        try:
            exp = explainer.explain_entry(entry)
            if exp:
                score["explanation"] = {
                    "top_features": exp["top_features"],
                    "confidence": exp["confidence"],
                    "prediction": exp["prediction"],
                }
        except Exception:
            pass

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

    if progress_cb:
        progress_cb(78, "Pushing scores to Seismo...")
    try:
        score_result = sync.push_scores(scores, model_info["version"], model_meta=model_meta)
    except Exception as e:
        detail = str(e)
        if isinstance(e, _httpx.HTTPStatusError):
            detail = "Seismo HTTP {}: {}".format(e.response.status_code, e.response.text[:300])
        raise ValueError("Failed to push scores: {}".format(detail))

    if progress_cb:
        progress_cb(88, "Pushing labels...")
    try:
        sync.push_labels()
    except Exception as e:
        logger.warning("Label push failed during score push: %s", e)
        db.log_sync("push", 0, "FAILED label push: {}".format(e))

    if progress_cb:
        progress_cb(93, "Building recipe...")
    try:
        recipe = distiller.distill_recipe()
    except Exception as e:
        logger.warning("Recipe distillation failed: %s", e)
        recipe = None

    recipe_result = {}
    if recipe:
        if progress_cb:
            progress_cb(97, "Pushing recipe...")
        try:
            recipe_result = sync.push_recipe(recipe)
        except Exception as e:
            detail = str(e)
            if isinstance(e, _httpx.HTTPStatusError):
                detail = "Seismo HTTP {}: {}".format(e.response.status_code, e.response.text[:300])
            recipe_result = {"error": detail}

    return {
        "success": True,
        "scores_pushed": len(scores),
        "score_result": score_result,
        "recipe_result": recipe_result,
    }


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


def _extract_legal_patterns(limit: int = 12) -> dict:
    """Read active recipe and expose top legal phrase patterns for dashboard UI."""
    model = db.get_active_model()
    if not model or not model.get("recipe_path"):
        return {"positive": [], "negative": []}
    recipe_path = Path(model["recipe_path"])
    if not recipe_path.exists():
        return {"positive": [], "negative": []}
    try:
        with open(recipe_path) as f:
            recipe = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"positive": [], "negative": []}

    keywords = recipe.get("keywords", {})
    if not keywords:
        return {"positive": [], "negative": []}

    # Focus on phrase-like features and legal-domain hints.
    legal_markers = (
        "eu", "eea", "ewr", "third", "country", "member state",
        "dritt", "tiers", "conformity", "assessment", "market access",
        "equivalence", "delegated", "implementing", "corrigendum",
        "annex", "regulation", "directive", "ce marking", "single market",
    )

    positives = []
    negatives = []
    for phrase, cls_wts in keywords.items():
        if " " not in phrase:
            continue
        p = phrase.lower()
        if not any(m in p for m in legal_markers):
            continue
        inv = float(cls_wts.get("investigation_lead", 0.0))
        imp = float(cls_wts.get("important", 0.0))
        noise = float(cls_wts.get("noise", 0.0))
        pos_score = inv + imp * 0.7
        neg_score = noise
        if pos_score > 0:
            positives.append((phrase, round(pos_score, 4)))
        if neg_score > 0:
            negatives.append((phrase, round(neg_score, 4)))

    positives.sort(key=lambda x: x[1], reverse=True)
    negatives.sort(key=lambda x: x[1], reverse=True)
    return {
        "positive": positives[:limit],
        "negative": negatives[:limit],
    }


# ─── Pages ───

@app.get("/", response_class=HTMLResponse)
async def labeling_page(request: Request, source: str = "all"):
    """Main labeling page — smart-sampled for active learning."""
    if not model_manager.has_profile():
        return RedirectResponse("/setup", status_code=302)
    ctx = _base_context(request)
    entry_type = None
    if source == "lex":
        entry_type = "lex_item"
    elif source == "news":
        entry_type = "feed_item"
    entries = sampler.get_smart_entries(limit=30, entry_type=entry_type)
    ctx["source_filter"] = source

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
    ctx["legal_patterns"] = _extract_legal_patterns()

    # Get keyword data if model exists
    ctx["keywords"] = {}
    if ctx["active_model"]:
        try:
            kw = explainer.global_keywords(limit=30)
            ctx["keywords"] = kw
        except Exception as e:
            logger.warning("Failed to load dashboard keywords: %s", e)

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
    except Exception as e:
        logger.warning("Background label push failed: %s", e)
        db.log_sync("push", 0, "FAILED: {}".format(e))

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
async def sync_pull(full: bool = False, background: bool = False):
    """Pull entries and labels from Seismo.

    full=False (default): quick sync (single pull + one embedding pass)
    full=True: full backfill by source type + embedding exhaustion pass
    """
    import asyncio
    if background:
        job_type = "sync_pull_full" if full else "sync_pull"
        job_id = _create_job(job_type)
        t = threading.Thread(
            target=lambda: _run_job(job_id, lambda cb: _sync_pull_impl(full=full, progress_cb=cb)),
            daemon=True,
        )
        t.start()
        return {"success": True, "job_id": job_id}

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _sync_pull_impl(full=full))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/sync/push")
async def sync_push(background: bool = False):
    """Score all entries and push scores + recipe to Seismo."""
    import asyncio
    if background:
        job_id = _create_job("sync_push")
        t = threading.Thread(
            target=lambda: _run_job(job_id, lambda cb: _sync_push_impl(progress_cb=cb)),
            daemon=True,
        )
        t.start()
        return {"success": True, "job_id": job_id}

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_push_impl)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    """Get background job status for sync/push progress polling."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/sync/labels")
async def sync_labels():
    """Push local labels to Seismo and pull any new labels back."""
    pushed = {}
    pulled = 0
    try:
        pushed = sync.push_labels()
    except Exception as e:
        raise HTTPException(500, f"Failed to push labels: {e}")
    pull_error = None
    try:
        pulled = sync.pull_labels()
    except Exception as e:
        logger.warning("Label pull failed: %s", e)
        pull_error = str(e)
    result = {"success": True, "pushed": pushed, "labels_imported": pulled}
    if pull_error:
        result["pull_error"] = pull_error
    return result


@app.get("/api/sync/test")
async def sync_test():
    """Test connection to Seismo, including a label endpoint smoke test."""
    ok, msg = sync.test_connection()
    if not ok:
        return {"success": False, "message": msg}
    # Also verify the label push endpoint works
    label_ok, label_msg = sync.verify_seismo_endpoints()
    if not label_ok:
        return {"success": True, "message": msg, "warning": label_msg}
    return {"success": True, "message": msg}


@app.get("/api/sync/health")
async def sync_health():
    """Return last sync status for each direction. Surfaces failures visibly."""
    syncs = db.get_recent_syncs(20)
    last_push = next((s for s in syncs if s["direction"] == "push"), None)
    last_pull = next((s for s in syncs if s["direction"] == "pull"), None)
    push_ok = last_push and not (last_push.get("details") or "").startswith("FAILED")
    pull_ok = last_pull and not (last_pull.get("details") or "").startswith("FAILED")
    return {
        "push": {"ok": push_ok, "detail": dict(last_push) if last_push else None},
        "pull": {"ok": pull_ok, "detail": dict(last_pull) if last_pull else None},
    }


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
    gpu_available = False
    try:
        import torch
        gpu_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        pass

    return {
        "magnitu_version": VERSION,
        "architecture": config.get("model_architecture", "transformer"),
        "entries": db.get_entry_count(),
        "labels": db.get_label_count(),
        "embeddings": db.get_embedding_count(),
        "label_distribution": db.get_label_distribution(),
        "active_model": model,
        "models_count": len(db.get_all_models()),
        "gpu_available": gpu_available,
        "gpu_enabled": config.get("use_gpu", True),
    }


# ─── API: Settings ───

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update configuration."""
    data = await request.json()
    config = get_config()
    old_transformer_name = config.get("transformer_model_name", "")
    old_use_gpu = config.get("use_gpu", True)

    for key in ["seismo_url", "api_key", "min_labels_to_train",
                "recipe_top_keywords", "auto_train_after_n_labels", "alert_threshold",
                "model_architecture", "transformer_model_name", "use_gpu"]:
        if key in data:
            config[key] = data[key]
    save_config(config)

    # If the transformer model name changed, invalidate all cached embeddings
    # and clear the in-memory model so the new one gets loaded on next use
    new_transformer_name = config.get("transformer_model_name", "")
    if new_transformer_name != old_transformer_name and old_transformer_name:
        db.invalidate_all_embeddings()
        pipeline.invalidate_embedder_cache()

    # Reload embedder on the new device when GPU setting changes
    if config.get("use_gpu", True) != old_use_gpu:
        pipeline.release_embedder()

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
