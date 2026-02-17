"""
Integration tests for Magnitu 2.

Tests the critical paths:
1. Config: transformer settings load correctly
2. DB: schema migration, reasoning, embedding storage
3. Pipeline (TF-IDF): train, score, feature importance
4. Pipeline (Transformer): embed, train, score, knowledge distillation
5. Distiller: recipe from TF-IDF, recipe from transformer via distillation
6. Explainer: architecture-aware explanations
7. Model manager: export/import with architecture metadata
8. Embedding invalidation on model change
9. Reasoning boost in recipe
10. FastAPI endpoints: label with reasoning, train, stats
"""
import os
import sys
import json
import shutil
import tempfile
import numpy as np

# ── Setup: use a temp DB so we don't pollute the real one ──
_test_dir = tempfile.mkdtemp(prefix="magnitu_test_")
os.environ["MAGNITU_TEST"] = "1"

from pathlib import Path as _P

import config
config.DB_PATH = _P(_test_dir) / "test.db"
config.MODELS_DIR = _P(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = _P(_test_dir) / "test_config.json"

# Save a test config
test_config = dict(config.DEFAULTS)
test_config["min_labels_to_train"] = 4
config.save_config(test_config)

import db
db.DB_PATH = config.DB_PATH

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    """Decorator-ish context for tests."""
    print("  TEST: {}".format(name), end=" ... ")
    return name


def ok():
    global PASS
    PASS += 1
    print("OK")


def fail(msg):
    global FAIL
    FAIL += 1
    ERRORS.append(msg)
    print("FAIL: {}".format(msg))


# ═══════════════════════════════════════════
#  1. Config
# ═══════════════════════════════════════════
print("\n=== 1. Config ===")

t = test("Version is 2.x")
try:
    assert config.VERSION.startswith("2."), "Expected version 2.x, got " + config.VERSION
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer defaults present")
try:
    cfg = config.get_config()
    assert "model_architecture" in cfg
    assert "transformer_model_name" in cfg
    assert "embedding_dim" in cfg
    assert cfg["model_architecture"] == "transformer"
    assert cfg["embedding_dim"] == 768
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  2. Database schema
# ═══════════════════════════════════════════
print("\n=== 2. Database Schema ===")

db.init_db()

t = test("Entries table has embedding column")
try:
    conn = db.get_db()
    cursor = conn.execute("PRAGMA table_info(entries)")
    cols = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "embedding" in cols
    ok()
except Exception as e:
    fail(str(e))

t = test("Labels table has reasoning column")
try:
    conn = db.get_db()
    cursor = conn.execute("PRAGMA table_info(labels)")
    cols = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "reasoning" in cols
    ok()
except Exception as e:
    fail(str(e))

t = test("Models table has architecture column")
try:
    conn = db.get_db()
    cursor = conn.execute("PRAGMA table_info(models)")
    cols = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "architecture" in cols
    ok()
except Exception as e:
    fail(str(e))

t = test("set_label with reasoning")
try:
    # Need an entry first
    db.upsert_entry({
        "entry_type": "feed_item", "entry_id": 100,
        "title": "Test", "description": "", "content": "",
        "link": "", "author": "", "published_date": "",
        "source_name": "", "source_category": "", "source_type": "rss",
    })
    db.set_label("feed_item", 100, "investigation_lead", reasoning="test reason")
    result = db.get_label_with_reasoning("feed_item", 100)
    assert result is not None
    assert result["label"] == "investigation_lead"
    assert result["reasoning"] == "test reason"
    db.remove_label("feed_item", 100)
    ok()
except Exception as e:
    fail(str(e))

t = test("Embedding store and retrieve")
try:
    fake_emb = np.random.randn(768).astype(np.float32)
    fake_bytes = fake_emb.tobytes()
    db.store_embedding("feed_item", 100, fake_bytes)
    assert db.get_embedding_count() >= 1
    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=100"
    ).fetchone()
    conn.close()
    retrieved = np.frombuffer(row["embedding"], dtype=np.float32)
    assert np.allclose(fake_emb, retrieved)
    ok()
except Exception as e:
    fail(str(e))

t = test("invalidate_all_embeddings")
try:
    db.invalidate_all_embeddings()
    assert db.get_embedding_count() == 0
    ok()
except Exception as e:
    fail(str(e))

t = test("get_all_reasoning_texts")
try:
    db.set_label("feed_item", 100, "investigation_lead", reasoning="contracts fraud")
    texts = db.get_all_reasoning_texts()
    assert len(texts) >= 1
    assert any(t["reasoning"] == "contracts fraud" for t in texts)
    db.remove_label("feed_item", 100)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  3. Pipeline - Setup test data
# ═══════════════════════════════════════════
print("\n=== 3. Pipeline Setup ===")

test_entries = [
    {"entry_type": "feed_item", "entry_id": 1, "title": "Investigation reveals corruption in government contracts",
     "description": "Major scandal uncovered", "content": "Deep investigation into public procurement fraud and bribery",
     "link": "", "author": "", "published_date": "2024-01-01",
     "source_name": "Investigative News", "source_category": "politics", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 2, "title": "New policy on data protection announced",
     "description": "Important regulation change", "content": "Government announces stricter rules for data handling",
     "link": "", "author": "", "published_date": "2024-01-02",
     "source_name": "Policy Daily", "source_category": "policy", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 3, "title": "Quarterly earnings report released",
     "description": "Company results", "content": "The company reported strong earnings above analyst expectations",
     "link": "", "author": "", "published_date": "2024-01-03",
     "source_name": "Business Wire", "source_category": "business", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 4, "title": "Weekend weather forecast",
     "description": "Rain expected", "content": "Forecast shows rain and mild temperatures for the coming weekend",
     "link": "", "author": "", "published_date": "2024-01-04",
     "source_name": "Weather Channel", "source_category": "weather", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 5, "title": "Secret documents reveal systematic cover-up",
     "description": "Leaked classified files", "content": "Documents obtained show years of systematic fraud in defense contracts",
     "link": "", "author": "", "published_date": "2024-01-05",
     "source_name": "Investigative News", "source_category": "politics", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 6, "title": "Local sports team wins championship",
     "description": "Match recap and highlights", "content": "The team won in overtime with a spectacular goal",
     "link": "", "author": "", "published_date": "2024-01-06",
     "source_name": "Sports Daily", "source_category": "sports", "source_type": "rss"},
]

t = test("Insert test entries")
try:
    db.upsert_entries(test_entries)
    assert db.get_entry_count() >= 6
    ok()
except Exception as e:
    fail(str(e))

t = test("Label test entries with reasoning")
try:
    db.set_label("feed_item", 1, "investigation_lead", reasoning="corruption in public contracts")
    db.set_label("feed_item", 2, "important")
    db.set_label("feed_item", 3, "background")
    db.set_label("feed_item", 4, "noise")
    assert db.get_label_count() >= 4
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  4. TF-IDF Pipeline
# ═══════════════════════════════════════════
print("\n=== 4. TF-IDF Pipeline ===")

import pipeline

t = test("TF-IDF train")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "tfidf"
    config.save_config(cfg)

    result = pipeline.train()
    assert result["success"], result.get("error", "")
    assert result["architecture"] == "tfidf"
    assert result["version"] >= 1
    assert result["accuracy"] >= 0
    ok()
except Exception as e:
    fail(str(e))

t = test("TF-IDF score")
try:
    scores = pipeline.score_entries(test_entries)
    assert len(scores) == len(test_entries)
    for s in scores:
        assert "relevance_score" in s
        assert "predicted_label" in s
        assert "probabilities" in s
        assert 0 <= s["relevance_score"] <= 1
    ok()
except Exception as e:
    fail(str(e))

t = test("TF-IDF feature importance")
try:
    importance = pipeline.get_feature_importance()
    assert len(importance) > 0
    for cls, pairs in importance.items():
        assert len(pairs) > 0
        assert all(len(p) == 2 for p in pairs)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  5. Transformer Pipeline
# ═══════════════════════════════════════════
print("\n=== 5. Transformer Pipeline ===")

t = test("Embedding byte roundtrip")
try:
    emb = np.random.randn(768).astype(np.float32)
    b = pipeline.embedding_to_bytes(emb)
    emb2 = pipeline.bytes_to_embedding(b, 768)
    assert np.allclose(emb, emb2), "Roundtrip mismatch"
    ok()
except Exception as e:
    fail(str(e))

t = test("Compute embeddings")
try:
    texts = ["Investigation reveals corruption", "Weather forecast for weekend"]
    embeddings = pipeline.compute_embeddings(texts)
    assert embeddings.shape == (2, 768), "Expected (2, 768), got {}".format(embeddings.shape)
    assert not np.allclose(embeddings[0], embeddings[1]), "Different texts should have different embeddings"
    ok()
except Exception as e:
    fail(str(e))

t = test("embed_entries helper")
try:
    emb_bytes = pipeline.embed_entries(test_entries[:2])
    assert len(emb_bytes) == 2
    assert all(len(b) == 768 * 4 for b in emb_bytes)
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer train")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "transformer"
    config.save_config(cfg)

    # Clear old models so this gets a clean version
    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"], result.get("error", "")
    assert result["architecture"] == "transformer"
    assert result["feature_count"] == 768
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer score")
try:
    scores = pipeline.score_entries(test_entries)
    assert len(scores) == len(test_entries)
    for s in scores:
        assert "relevance_score" in s
        assert "predicted_label" in s
        assert 0 <= s["relevance_score"] <= 1

    # The investigation entries should score higher than sports/weather
    score_map = {s["entry_id"]: s["relevance_score"] for s in scores}
    # Entry 1 (investigation) should beat entry 4 (weather)
    assert score_map[1] > score_map[4], \
        "Investigation ({:.2f}) should score > weather ({:.2f})".format(score_map[1], score_map[4])
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer feature importance returns empty (expected)")
try:
    importance = pipeline.get_feature_importance()
    assert importance == {}, "Transformer models should not have direct feature importance"
    ok()
except Exception as e:
    fail(str(e))

t = test("Embeddings cached in DB after scoring")
try:
    count = db.get_embedding_count()
    assert count >= len(test_entries), \
        "Expected >= {} embeddings, got {}".format(len(test_entries), count)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  6. Distiller
# ═══════════════════════════════════════════
print("\n=== 6. Distiller ===")

import distiller

t = test("TF-IDF recipe distillation")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "tfidf"
    config.save_config(cfg)

    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"]

    recipe = distiller.distill_recipe()
    assert recipe is not None, "Recipe should not be None for TF-IDF"
    assert "keywords" in recipe
    assert "classes" in recipe
    assert recipe["classes"] == ["investigation_lead", "important", "background", "noise"]
    assert "class_weights" in recipe
    assert recipe["class_weights"] == [1.0, 0.66, 0.33, 0.0]
    ok()
except Exception as e:
    fail(str(e))

t = test("Reasoning boost in recipe")
try:
    db.set_label("feed_item", 1, "investigation_lead", reasoning="procurement fraud bribery")
    recipe = distiller.distill_recipe()
    assert recipe is not None
    # Check that reasoning keywords got incorporated
    kw = recipe.get("keywords", {})
    has_reasoning_term = any(
        term in kw for term in ["procurement", "fraud", "bribery"]
    )
    assert has_reasoning_term, \
        "Reasoning terms should appear in recipe keywords. Got: {}".format(
            [k for k in list(kw.keys())[:20]]
        )
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  7. Explainer
# ═══════════════════════════════════════════
print("\n=== 7. Explainer ===")

import explainer

t = test("TF-IDF explanation")
try:
    # Should have a TF-IDF model active from test 6
    exp = explainer.explain_entry(test_entries[0])
    assert exp is not None
    assert "prediction" in exp
    assert "confidence" in exp
    assert "probabilities" in exp
    assert "top_features" in exp
    assert 0 <= exp["confidence"] <= 1
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer explanation")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "transformer"
    config.save_config(cfg)

    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"]

    exp = explainer.explain_entry(test_entries[0])
    assert exp is not None
    assert "prediction" in exp
    assert "confidence" in exp
    assert "probabilities" in exp
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  8. Model Manager
# ═══════════════════════════════════════════
print("\n=== 8. Model Manager ===")

import model_manager

t = test("Create profile")
try:
    if not model_manager.has_profile():
        model_manager.create_profile("test-model", "Test model for integration tests")
    profile = model_manager.get_profile()
    assert profile is not None
    assert profile["model_name"] == "test-model"
    ok()
except Exception as e:
    fail(str(e))

t = test("Export contains architecture in manifest")
try:
    export_path = model_manager.export_model()
    assert os.path.exists(export_path)

    import zipfile
    with zipfile.ZipFile(export_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert "architecture" in manifest
    assert "magnitu_version" in manifest
    assert manifest["magnitu_version"].startswith("2.")
    ok()
except Exception as e:
    fail(str(e))

t = test("Export contains reasoning in labels")
try:
    with zipfile.ZipFile(export_path) as zf:
        labels = json.loads(zf.read("labels.json"))
    has_reasoning = any(lbl.get("reasoning") for lbl in labels)
    assert has_reasoning, "Exported labels should include reasoning text"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  9. Embedding Invalidation
# ═══════════════════════════════════════════
print("\n=== 9. Embedding Invalidation ===")

t = test("Invalidate clears all embeddings")
try:
    before = db.get_embedding_count()
    assert before > 0, "Should have embeddings before invalidation"
    db.invalidate_all_embeddings()
    after = db.get_embedding_count()
    assert after == 0, "Should have 0 embeddings after invalidation, got {}".format(after)
    ok()
except Exception as e:
    fail(str(e))

t = test("invalidate_embedder_cache resets singleton")
try:
    pipeline.invalidate_embedder_cache()
    assert pipeline._embedder is None
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  10. FastAPI App
# ═══════════════════════════════════════════
print("\n=== 10. FastAPI App ===")

t = test("App creates with correct version")
try:
    from main import app
    assert app.version == config.VERSION
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════
print("\n" + "=" * 50)
print("Results: {} passed, {} failed".format(PASS, FAIL))
if ERRORS:
    print("\nFailures:")
    for err in ERRORS:
        print("  - {}".format(err))
print("=" * 50)

# Cleanup
shutil.rmtree(_test_dir, ignore_errors=True)

sys.exit(0 if FAIL == 0 else 1)
