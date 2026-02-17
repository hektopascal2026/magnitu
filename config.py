"""
Magnitu configuration.
Loaded from environment variables or magnitu_config.json.
"""
import os
import json
from pathlib import Path

VERSION = "2.0.0"

BASE_DIR = Path(__file__).parent

CONFIG_PATH = BASE_DIR / "magnitu_config.json"
DB_PATH = BASE_DIR / "magnitu.db"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Defaults
DEFAULTS = {
    "seismo_url": "http://localhost/seismo_0.3/index.php",
    "api_key": "",
    "min_labels_to_train": 20,
    "recipe_top_keywords": 200,
    "auto_train_after_n_labels": 10,
    "alert_threshold": 0.75,
    # Transformer settings (Magnitu 2)
    "model_architecture": "transformer",     # "tfidf" or "transformer"
    "transformer_model_name": "distilroberta-base",
    "embedding_dim": 768,
}


def load_config() -> dict:
    """Load config from JSON file, merging with defaults."""
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            config.update(saved)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def save_config(config: dict):
    """Save config to JSON file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_config() -> dict:
    """Get current config (cached in module)."""
    return load_config()
