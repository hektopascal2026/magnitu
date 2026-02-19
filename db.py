"""
Local SQLite database for Magnitu.
Stores cached entries, user labels, model history, and sync log.
"""
import sqlite3
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
from config import DB_PATH


def get_db() -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_db(conn: sqlite3.Connection):
    """Run migrations for Magnitu 2 schema additions."""
    cursor = conn.execute("PRAGMA table_info(entries)")
    entry_cols = {row[1] for row in cursor.fetchall()}

    if "embedding" not in entry_cols:
        conn.execute("ALTER TABLE entries ADD COLUMN embedding BLOB DEFAULT NULL")

    cursor = conn.execute("PRAGMA table_info(labels)")
    label_cols = {row[1] for row in cursor.fetchall()}

    if "reasoning" not in label_cols:
        conn.execute("ALTER TABLE labels ADD COLUMN reasoning TEXT DEFAULT ''")

    cursor = conn.execute("PRAGMA table_info(models)")
    model_cols = {row[1] for row in cursor.fetchall()}

    if "architecture" not in model_cols:
        conn.execute("ALTER TABLE models ADD COLUMN architecture TEXT DEFAULT 'tfidf'")

    conn.commit()


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_type TEXT NOT NULL,        -- 'feed_item', 'email', 'lex_item'
            entry_id INTEGER NOT NULL,       -- ID in seismo's database
            title TEXT DEFAULT '',
            description TEXT DEFAULT '',
            content TEXT DEFAULT '',
            link TEXT DEFAULT '',
            author TEXT DEFAULT '',
            published_date TEXT DEFAULT '',
            source_name TEXT DEFAULT '',
            source_category TEXT DEFAULT '',
            source_type TEXT DEFAULT '',      -- 'rss', 'substack', 'email', 'lex_eu', 'lex_ch'
            embedding BLOB DEFAULT NULL,     -- cached transformer embedding (Magnitu 2)
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entry_type, entry_id)
        );

        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_type TEXT NOT NULL,
            entry_id INTEGER NOT NULL,
            label TEXT NOT NULL,             -- 'investigation_lead', 'important', 'background', 'noise'
            reasoning TEXT DEFAULT '',       -- optional free-text explanation (Magnitu 2)
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entry_type, entry_id)
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            accuracy REAL DEFAULT 0.0,
            f1_score REAL DEFAULT 0.0,
            precision_score REAL DEFAULT 0.0,
            recall_score REAL DEFAULT 0.0,
            label_count INTEGER DEFAULT 0,
            label_distribution TEXT DEFAULT '{}',   -- JSON
            feature_count INTEGER DEFAULT 0,
            model_path TEXT DEFAULT '',
            recipe_path TEXT DEFAULT '',
            recipe_quality REAL DEFAULT 0.0,        -- how well recipe approximates full model
            architecture TEXT DEFAULT 'tfidf',      -- 'tfidf' or 'transformer' (Magnitu 2)
            trained_at TEXT DEFAULT (datetime('now')),
            is_active INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            direction TEXT NOT NULL,         -- 'pull' or 'push'
            items_count INTEGER DEFAULT 0,
            details TEXT DEFAULT '',
            synced_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_profile (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_uuid TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_entries_type_id ON entries(entry_type, entry_id);
        CREATE INDEX IF NOT EXISTS idx_labels_type_id ON labels(entry_type, entry_id);
        CREATE INDEX IF NOT EXISTS idx_labels_label ON labels(label);
        CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
    """)
    _migrate_db(conn)
    conn.commit()
    conn.close()


# ─── Entry operations ───

def upsert_entry(entry: dict):
    """Insert or update a cached entry from seismo.
    Invalidates cached embedding when text content changes so it gets recomputed."""
    conn = get_db()
    conn.execute("""
        INSERT INTO entries (entry_type, entry_id, title, description, content, link, author,
                            published_date, source_name, source_category, source_type)
        VALUES (:entry_type, :entry_id, :title, :description, :content, :link, :author,
                :published_date, :source_name, :source_category, :source_type)
        ON CONFLICT(entry_type, entry_id) DO UPDATE SET
            title=excluded.title, description=excluded.description, content=excluded.content,
            link=excluded.link, author=excluded.author, published_date=excluded.published_date,
            source_name=excluded.source_name, source_category=excluded.source_category,
            source_type=excluded.source_type, fetched_at=datetime('now'),
            embedding = CASE
                WHEN entries.title != excluded.title
                  OR entries.description != excluded.description
                  OR entries.content != excluded.content
                THEN NULL
                ELSE entries.embedding
            END
    """, entry)
    conn.commit()
    conn.close()


def upsert_entries(entries: List[dict]):
    """Batch upsert entries.
    Invalidates cached embedding when text content changes so it gets recomputed."""
    conn = get_db()
    conn.executemany("""
        INSERT INTO entries (entry_type, entry_id, title, description, content, link, author,
                            published_date, source_name, source_category, source_type)
        VALUES (:entry_type, :entry_id, :title, :description, :content, :link, :author,
                :published_date, :source_name, :source_category, :source_type)
        ON CONFLICT(entry_type, entry_id) DO UPDATE SET
            title=excluded.title, description=excluded.description, content=excluded.content,
            link=excluded.link, author=excluded.author, published_date=excluded.published_date,
            source_name=excluded.source_name, source_category=excluded.source_category,
            source_type=excluded.source_type, fetched_at=datetime('now'),
            embedding = CASE
                WHEN entries.title != excluded.title
                  OR entries.description != excluded.description
                  OR entries.content != excluded.content
                THEN NULL
                ELSE entries.embedding
            END
    """, entries)
    conn.commit()
    conn.close()


def get_unlabeled_entries(limit: int = 30) -> List[dict]:
    """Get entries that haven't been labeled yet, newest first."""
    conn = get_db()
    rows = conn.execute("""
        SELECT e.* FROM entries e
        LEFT JOIN labels l ON e.entry_type = l.entry_type AND e.entry_id = l.entry_id
        WHERE l.id IS NULL
        ORDER BY e.published_date DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_entries() -> List[dict]:
    """Get all cached entries."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM entries ORDER BY published_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_entries(days: int = 7) -> List[dict]:
    """Get entries published within the last N days."""
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM entries
        WHERE published_date >= date('now', ?)
        ORDER BY published_date DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_labeled_entries() -> List[dict]:
    """Get all entries that have a user label, with the label included."""
    conn = get_db()
    rows = conn.execute("""
        SELECT e.*, l.label as user_label
        FROM entries e
        JOIN labels l ON e.entry_type = l.entry_type AND e.entry_id = l.entry_id
        ORDER BY e.published_date DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_entry_count() -> int:
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    conn.close()
    return count


# ─── Label operations ───

def set_label(entry_type: str, entry_id: int, label: str, reasoning: str = ""):
    """Set or update a label for an entry, with optional reasoning text."""
    conn = get_db()
    conn.execute("""
        INSERT INTO labels (entry_type, entry_id, label, reasoning)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(entry_type, entry_id) DO UPDATE SET
            label=excluded.label, reasoning=excluded.reasoning, updated_at=datetime('now')
    """, (entry_type, entry_id, label, reasoning))
    conn.commit()
    conn.close()


def remove_label(entry_type: str, entry_id: int):
    """Remove a label."""
    conn = get_db()
    conn.execute("DELETE FROM labels WHERE entry_type = ? AND entry_id = ?", (entry_type, entry_id))
    conn.commit()
    conn.close()


def get_label(entry_type: str, entry_id: int) -> Optional[str]:
    """Get label for a specific entry."""
    conn = get_db()
    row = conn.execute(
        "SELECT label FROM labels WHERE entry_type = ? AND entry_id = ?",
        (entry_type, entry_id)
    ).fetchone()
    conn.close()
    return row["label"] if row else None


def get_label_with_reasoning(entry_type: str, entry_id: int) -> Optional[dict]:
    """Get label and reasoning for a specific entry."""
    conn = get_db()
    row = conn.execute(
        "SELECT label, reasoning FROM labels WHERE entry_type = ? AND entry_id = ?",
        (entry_type, entry_id)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"label": row["label"], "reasoning": row["reasoning"] or ""}


def get_all_labels() -> List[dict]:
    """Get all labels with entry data."""
    conn = get_db()
    rows = conn.execute("""
        SELECT l.entry_type, l.entry_id, l.label, l.reasoning, l.created_at, l.updated_at,
               e.title, e.description, e.content, e.source_type, e.source_name, e.source_category
        FROM labels l
        JOIN entries e ON l.entry_type = e.entry_type AND l.entry_id = e.entry_id
        ORDER BY l.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_labels_raw() -> List[dict]:
    """Get all labels without joining entries (for syncing)."""
    conn = get_db()
    rows = conn.execute("""
        SELECT entry_type, entry_id, label, reasoning, created_at, updated_at
        FROM labels
        ORDER BY updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_label_count() -> int:
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    conn.close()
    return count


def get_label_distribution() -> dict:
    """Get count of labels per class."""
    conn = get_db()
    rows = conn.execute(
        "SELECT label, COUNT(*) as count FROM labels GROUP BY label ORDER BY count DESC"
    ).fetchall()
    conn.close()
    return {r["label"]: r["count"] for r in rows}


def get_labels_since_model(model_version: int) -> int:
    """Count labels added since the last model was trained."""
    conn = get_db()
    # Get the trained_at of the given model version
    model_row = conn.execute(
        "SELECT trained_at FROM models WHERE version = ?", (model_version,)
    ).fetchone()
    if not model_row:
        count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    else:
        count = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE updated_at > ?", (model_row["trained_at"],)
        ).fetchone()[0]
    conn.close()
    return count


# ─── Embedding operations (Magnitu 2) ───

def store_embedding(entry_type: str, entry_id: int, embedding_bytes: bytes):
    """Store a precomputed embedding for an entry."""
    conn = get_db()
    conn.execute("""
        UPDATE entries SET embedding = ? WHERE entry_type = ? AND entry_id = ?
    """, (embedding_bytes, entry_type, entry_id))
    conn.commit()
    conn.close()


def store_embeddings_batch(updates: List[tuple]):
    """Batch store embeddings. Each tuple: (embedding_bytes, entry_type, entry_id)."""
    conn = get_db()
    conn.executemany("""
        UPDATE entries SET embedding = ? WHERE entry_type = ? AND entry_id = ?
    """, updates)
    conn.commit()
    conn.close()


def get_entries_without_embeddings(limit: int = 500) -> List[dict]:
    """Get entries that don't have embeddings yet."""
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM entries WHERE embedding IS NULL
        ORDER BY published_date DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_embedding_count() -> int:
    """Count entries that have embeddings."""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL").fetchone()[0]
    conn.close()
    return count


def get_all_reasoning_texts() -> List[dict]:
    """Get all labels that have reasoning text, for recipe boosting."""
    conn = get_db()
    rows = conn.execute("""
        SELECT entry_type, entry_id, label, reasoning
        FROM labels
        WHERE reasoning IS NOT NULL AND reasoning != ''
        ORDER BY updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def invalidate_all_embeddings():
    """Clear all cached embeddings (needed when transformer model changes)."""
    conn = get_db()
    conn.execute("UPDATE entries SET embedding = NULL")
    conn.commit()
    conn.close()


# ─── Model operations ───

def save_model_record(version: int, accuracy: float, f1: float, precision: float,
                      recall: float, label_count: int, label_dist: dict,
                      feature_count: int, model_path: str, recipe_path: str = "",
                      recipe_quality: float = 0.0) -> int:
    """Save a model training record and set it as active."""
    conn = get_db()
    # Deactivate all existing models
    conn.execute("UPDATE models SET is_active = 0")
    conn.execute("""
        INSERT INTO models (version, accuracy, f1_score, precision_score, recall_score,
                           label_count, label_distribution, feature_count, model_path,
                           recipe_path, recipe_quality, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """, (version, accuracy, f1, precision, recall, label_count,
          json.dumps(label_dist), feature_count, model_path, recipe_path, recipe_quality))
    model_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return model_id


def get_active_model() -> Optional[dict]:
    """Get the currently active model record."""
    conn = get_db()
    row = conn.execute("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_models() -> List[dict]:
    """Get all model records, newest first."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM models ORDER BY version DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_next_model_version() -> int:
    """Get the next model version number."""
    conn = get_db()
    row = conn.execute("SELECT MAX(version) FROM models").fetchone()
    conn.close()
    return (row[0] or 0) + 1


# ─── Sync log ───

def log_sync(direction: str, items_count: int, details: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO sync_log (direction, items_count, details) VALUES (?, ?, ?)",
        (direction, items_count, details)
    )
    conn.commit()
    conn.close()


def get_recent_syncs(limit: int = 20) -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM sync_log ORDER BY synced_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Model profile ───

def get_model_profile() -> Optional[dict]:
    """Get the current model profile (only one exists at a time)."""
    conn = get_db()
    row = conn.execute("SELECT * FROM model_profile ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def set_model_profile(model_name: str, model_uuid: str, description: str = "",
                      created_at: str = None):
    """Set the model profile (replaces any existing profile)."""
    conn = get_db()
    conn.execute("DELETE FROM model_profile")
    conn.execute("""
        INSERT INTO model_profile (model_name, model_uuid, description, created_at)
        VALUES (?, ?, ?, ?)
    """, (model_name, model_uuid, description,
          created_at or datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def update_model_profile(description: str = None):
    """Update description of the current profile. Name is immutable once set."""
    profile = get_model_profile()
    if not profile:
        return
    conn = get_db()
    if description is not None:
        conn.execute("UPDATE model_profile SET description = ?", (description,))
    conn.commit()
    conn.close()


def has_model_profile() -> bool:
    """Quick check if a model profile exists."""
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) FROM model_profile").fetchone()
    conn.close()
    return row[0] > 0


def export_labels() -> List[dict]:
    """Export all labels with their entry text (for .magnitu package).
    Returns list of dicts with label + entry data needed for retraining."""
    conn = get_db()
    rows = conn.execute("""
        SELECT l.entry_type, l.entry_id, l.label, l.reasoning, l.created_at, l.updated_at,
               e.title, e.description, e.content, e.link, e.author,
               e.published_date, e.source_name, e.source_category, e.source_type
        FROM labels l
        LEFT JOIN entries e ON l.entry_type = e.entry_type AND l.entry_id = e.entry_id
        ORDER BY l.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def import_labels(labels_list: List[dict]) -> dict:
    """Import labels from a .magnitu package. Merges with existing (newer wins).
    Also upserts the associated entry data so retraining works.
    Returns {imported, skipped, updated}."""
    conn = get_db()
    imported = 0
    skipped = 0
    updated = 0

    # Use explicit transaction for performance
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Pre-fetch all existing labels for fast lookup
        existing_map = {}
        for row in conn.execute("SELECT entry_type, entry_id, label, updated_at FROM labels"):
            existing_map[(row["entry_type"], row["entry_id"])] = {
                "label": row["label"], "updated_at": row["updated_at"] or ""
            }

        for lbl in labels_list:
            entry_type = lbl.get("entry_type", "")
            entry_id = lbl.get("entry_id")
            label = lbl.get("label", "")
            lbl_updated_at = lbl.get("updated_at", "")

            if not entry_type or not entry_id or not label:
                skipped += 1
                continue

            key = (entry_type, entry_id)
            existing = existing_map.get(key)

            reasoning = lbl.get("reasoning", "")

            if existing:
                if lbl_updated_at > existing["updated_at"]:
                    conn.execute("""
                        UPDATE labels SET label = ?, reasoning = ?, updated_at = ?
                        WHERE entry_type = ? AND entry_id = ?
                    """, (label, reasoning, lbl_updated_at, entry_type, entry_id))
                    updated += 1
                else:
                    skipped += 1
            else:
                conn.execute("""
                    INSERT INTO labels (entry_type, entry_id, label, reasoning, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (entry_type, entry_id, label, reasoning,
                      lbl.get("created_at", ""), lbl_updated_at))
                imported += 1

            # Upsert entry data if present
            if lbl.get("title") is not None:
                conn.execute("""
                    INSERT INTO entries (entry_type, entry_id, title, description, content,
                                        link, author, published_date, source_name,
                                        source_category, source_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entry_type, entry_id) DO UPDATE SET
                        title=excluded.title, description=excluded.description,
                        content=excluded.content, link=excluded.link, author=excluded.author,
                        published_date=excluded.published_date, source_name=excluded.source_name,
                        source_category=excluded.source_category, source_type=excluded.source_type
                """, (entry_type, entry_id, lbl.get("title", ""),
                      lbl.get("description", ""), lbl.get("content", ""),
                      lbl.get("link", ""), lbl.get("author", ""),
                      lbl.get("published_date", ""), lbl.get("source_name", ""),
                      lbl.get("source_category", ""), lbl.get("source_type", "")))

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return {"imported": imported, "skipped": skipped, "updated": updated}


# Initialize on import
init_db()
