"""
Sync engine: connects to Seismo's API to fetch entries and push scores/recipe.

Magnitu 2: after pulling entries, computes transformer embeddings for new
entries and stores them in the DB so scoring stays instant.
"""
import logging
import httpx
from config import get_config
import db

logger = logging.getLogger(__name__)


def _request(method: str, params: dict, **kwargs) -> httpx.Response:
    """Make a request to Seismo with auth. Avoids base_url trailing-slash issues."""
    cfg = get_config()
    url = cfg["seismo_url"]
    params["api_key"] = cfg["api_key"]
    with httpx.Client(timeout=30.0) as client:
        resp = client.request(method, url, params=params, **kwargs)
        resp.raise_for_status()
        return resp


def pull_entries(since: str = None, entry_type: str = "all", limit: int = 500) -> int:
    """
    Fetch entries from Seismo and store locally.
    In transformer mode, also computes embeddings for new entries.
    Returns number of entries fetched.
    """
    params = {"action": "magnitu_entries", "type": entry_type, "limit": str(limit)}
    if since:
        params["since"] = since

    data = _request("GET", params).json()

    entries = data.get("entries", [])
    if entries:
        db.upsert_entries(entries)
        db.log_sync("pull", len(entries), "type={}, since={}".format(entry_type, since))

    # Compute embeddings for entries that don't have them yet
    cfg = get_config()
    if cfg.get("model_architecture") == "transformer":
        _compute_pending_embeddings()

    return len(entries)


def _compute_pending_embeddings():
    """Compute and store embeddings for all entries that lack them."""
    unembedded = db.get_entries_without_embeddings(limit=1000)
    if not unembedded:
        return

    logger.info("Computing embeddings for %d entries...", len(unembedded))
    try:
        from pipeline import embed_entries
        emb_bytes_list = embed_entries(unembedded)
        updates = []
        for entry, emb_bytes in zip(unembedded, emb_bytes_list):
            updates.append((emb_bytes, entry["entry_type"], entry["entry_id"]))
        db.store_embeddings_batch(updates)
        logger.info("Stored %d embeddings.", len(updates))
    except Exception as e:
        logger.warning("Failed to compute embeddings: %s", e)


def push_scores(scores: list[dict], model_version: int, model_meta: dict = None) -> dict:
    """
    Push batch of scores to Seismo.
    Each score: {entry_type, entry_id, relevance_score, predicted_label, explanation}
    model_meta: optional {model_name, model_description, model_version, model_trained_at}
    """
    payload = {
        "scores": scores,
        "model_version": model_version,
    }
    if model_meta:
        payload["model_meta"] = model_meta

    result = _request("POST", {"action": "magnitu_scores"}, json=payload).json()
    db.log_sync("push", len(scores), f"scores pushed, model v{model_version}")
    return result


def push_recipe(recipe: dict) -> dict:
    """Push a scoring recipe to Seismo."""
    result = _request("POST", {"action": "magnitu_recipe"}, json=recipe).json()
    db.log_sync("push", 1, f"recipe v{recipe.get('version', '?')} pushed")
    return result


def push_labels() -> dict:
    """Push all local labels to Seismo so other instances can pull them."""
    all_labels = db.get_all_labels_raw()
    if not all_labels:
        return {"success": True, "pushed": 0}

    payload = {
        "labels": [
            {
                "entry_type": lbl["entry_type"],
                "entry_id": lbl["entry_id"],
                "label": lbl["label"],
                "reasoning": lbl.get("reasoning", ""),
                "labeled_at": lbl.get("updated_at") or lbl.get("created_at", ""),
            }
            for lbl in all_labels
        ]
    }

    result = _request("POST", {"action": "magnitu_labels"}, json=payload).json()
    db.log_sync("push", len(all_labels), "labels pushed")
    return result


def pull_labels() -> int:
    """Pull labels from Seismo and merge into local database. Returns count imported."""
    data = _request("GET", {"action": "magnitu_labels"}).json()
    labels = data.get("labels", [])

    imported = 0
    for lbl in labels:
        entry_type = lbl.get("entry_type", "")
        entry_id = int(lbl.get("entry_id", 0))
        label = lbl.get("label", "")
        if not entry_type or not entry_id or not label:
            continue

        # Only import if we don't already have a label for this entry
        existing = db.get_label(entry_type, entry_id)
        if existing is None:
            db.set_label(entry_type, entry_id, label)
            imported += 1

    if imported:
        db.log_sync("pull", imported, "labels pulled from Seismo")
    return imported


def get_status() -> dict:
    """Check Seismo connectivity and status."""
    return _request("GET", {"action": "magnitu_status"}).json()


def test_connection() -> tuple[bool, str]:
    """Test if we can connect to Seismo. Returns (success, message)."""
    try:
        status = get_status()
        if status.get("status") == "ok":
            total = status.get("entries", {}).get("total", 0)
            return True, f"Connected. Seismo has {total} entries."
        return False, f"Unexpected response: {status}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return False, "Authentication failed. Check your API key."
        return False, f"HTTP error {e.response.status_code}: {e.response.text}"
    except httpx.ConnectError:
        return False, "Connection failed. Check the Seismo URL."
    except Exception as e:
        return False, f"Error: {str(e)}"
