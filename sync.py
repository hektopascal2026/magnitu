"""
Sync engine: connects to Seismo's API to fetch entries and push scores/recipe.
"""
import httpx
from config import get_config
import db


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
    Returns number of entries fetched.
    """
    params = {"action": "magnitu_entries", "type": entry_type, "limit": str(limit)}
    if since:
        params["since"] = since

    data = _request("GET", params).json()

    entries = data.get("entries", [])
    if entries:
        db.upsert_entries(entries)
        db.log_sync("pull", len(entries), f"type={entry_type}, since={since}")

    return len(entries)


def push_scores(scores: list[dict], model_version: int) -> dict:
    """
    Push batch of scores to Seismo.
    Each score: {entry_type, entry_id, relevance_score, predicted_label, explanation}
    """
    payload = {
        "scores": scores,
        "model_version": model_version,
    }

    result = _request("POST", {"action": "magnitu_scores"}, json=payload).json()
    db.log_sync("push", len(scores), f"scores pushed, model v{model_version}")
    return result


def push_recipe(recipe: dict) -> dict:
    """Push a scoring recipe to Seismo."""
    result = _request("POST", {"action": "magnitu_recipe"}, json=recipe).json()
    db.log_sync("push", 1, f"recipe v{recipe.get('version', '?')} pushed")
    return result


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
