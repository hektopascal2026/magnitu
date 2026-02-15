"""
Model Manager: handles export/import of portable .magnitu model packages.

A .magnitu file is a zip archive containing:
  - manifest.json  — model identity, version chain, metrics
  - model.joblib   — trained sklearn pipeline (if trained)
  - recipe.json    — distilled recipe (if exists)
  - labels.json    — all labeled entries with text (for retraining)
"""
import json
import uuid
import shutil
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib

import db
from config import MODELS_DIR


# ─── Profile helpers ───

def has_profile() -> bool:
    """Check if a model profile exists."""
    return db.has_model_profile()


def get_profile() -> Optional[dict]:
    """Get the current model profile or None."""
    return db.get_model_profile()


def create_profile(name: str, description: str = "") -> dict:
    """Create a new model profile with a fresh UUID."""
    model_uuid = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    db.set_model_profile(
        model_name=name,
        model_uuid=model_uuid,
        description=description,
        created_at=now,
    )
    return {
        "model_name": name,
        "model_uuid": model_uuid,
        "description": description,
        "created_at": now,
    }


def update_profile(name: str = None, description: str = None):
    """Update name and/or description of current profile."""
    db.update_model_profile(model_name=name, description=description)


# ─── Version chain ───

def _get_version_chain() -> list:
    """Build version chain from model training history."""
    models = db.get_all_models()
    chain = []
    for m in reversed(models):  # oldest first
        chain.append({
            "version": m["version"],
            "trained_at": m.get("trained_at", ""),
            "label_count": m.get("label_count", 0),
            "accuracy": m.get("accuracy", 0.0),
        })
    return chain


# ─── Export ───

def export_model(output_path: str = None) -> str:
    """
    Export current model as a .magnitu zip package.
    Returns the path to the created file.
    """
    profile = get_profile()
    if not profile:
        raise ValueError("No model profile configured. Create one first.")

    active_model = db.get_active_model()
    version_chain = _get_version_chain()

    # Build manifest
    manifest = {
        "model_name": profile["model_name"],
        "model_uuid": profile["model_uuid"],
        "description": profile.get("description", ""),
        "created_at": profile.get("created_at", ""),
        "exported_at": datetime.utcnow().isoformat(),
        "version": active_model["version"] if active_model else 0,
        "label_count": db.get_label_count(),
        "metrics": {},
        "version_chain": version_chain,
    }

    if active_model:
        manifest["metrics"] = {
            "accuracy": active_model.get("accuracy", 0.0),
            "f1": active_model.get("f1_score", 0.0),
            "precision": active_model.get("precision_score", 0.0),
            "recall": active_model.get("recall_score", 0.0),
        }

    # Export labels with entry data
    labels = db.export_labels()

    # Default output path
    if not output_path:
        safe_name = profile["model_name"].replace(" ", "_").lower()
        output_path = str(MODELS_DIR / f"{safe_name}.magnitu")

    # Create zip
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Write manifest
        with open(tmp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Write labels
        with open(tmp_dir / "labels.json", "w") as f:
            json.dump(labels, f, ensure_ascii=False)

        # Copy model.joblib if exists
        if active_model and active_model.get("model_path"):
            model_path = Path(active_model["model_path"])
            if model_path.exists():
                shutil.copy2(str(model_path), str(tmp_dir / "model.joblib"))

        # Copy recipe if exists
        if active_model and active_model.get("recipe_path"):
            recipe_path = Path(active_model["recipe_path"])
            if recipe_path.exists():
                shutil.copy2(str(recipe_path), str(tmp_dir / "recipe.json"))

        # Zip it
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in tmp_dir.iterdir():
                zf.write(str(file_path), file_path.name)

    return output_path


# ─── Import ───

def import_model(file_path: str) -> dict:
    """
    Import a .magnitu package. Merges labels (newer wins).
    Loads the trained model only if its version is higher than the local one.

    Returns dict with import results.
    """
    if not zipfile.is_zipfile(file_path):
        raise ValueError("Not a valid .magnitu file (not a zip archive).")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Extract
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(str(tmp_dir))

        # Read manifest
        manifest_path = tmp_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Invalid .magnitu file: no manifest.json found.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        imported_name = manifest.get("model_name", "Unknown")
        imported_uuid = manifest.get("model_uuid", "")
        imported_version = manifest.get("version", 0)
        imported_description = manifest.get("description", "")

        result = {
            "model_name": imported_name,
            "model_uuid": imported_uuid,
            "imported_version": imported_version,
            "labels": {"imported": 0, "skipped": 0, "updated": 0},
            "model_loaded": False,
            "message": "",
        }

        # Import labels
        labels_path = tmp_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                labels = json.load(f)
            result["labels"] = db.import_labels(labels)

        # Determine if we should load the trained model
        local_model = db.get_active_model()
        local_version = local_model["version"] if local_model else 0

        model_file = tmp_dir / "model.joblib"
        should_load_model = False

        if model_file.exists():
            if local_version == 0:
                # No local model — always load
                should_load_model = True
            elif imported_version > local_version:
                # Imported is newer — load it
                should_load_model = True

        if should_load_model and model_file.exists():
            # Copy model to models dir
            dest = MODELS_DIR / f"model_v{imported_version}.joblib"
            shutil.copy2(str(model_file), str(dest))

            # Copy recipe if present
            recipe_file = tmp_dir / "recipe.json"
            recipe_dest = ""
            if recipe_file.exists():
                recipe_dest = str(MODELS_DIR / f"recipe_v{imported_version}.json")
                shutil.copy2(str(recipe_file), recipe_dest)

            # Get metrics from manifest
            metrics = manifest.get("metrics", {})

            # Save model record in DB
            db.save_model_record(
                version=imported_version,
                accuracy=metrics.get("accuracy", 0.0),
                f1=metrics.get("f1", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                label_count=manifest.get("label_count", 0),
                label_dist={},
                feature_count=0,
                model_path=str(dest),
                recipe_path=recipe_dest,
            )
            result["model_loaded"] = True

        # Set or update model profile
        current_profile = get_profile()
        if not current_profile:
            # No profile yet — adopt the imported one
            db.set_model_profile(
                model_name=imported_name,
                model_uuid=imported_uuid,
                description=imported_description,
                created_at=manifest.get("created_at", ""),
            )
        else:
            # Profile exists — keep it but note the import
            pass

        # Build message
        new_labels = result["labels"]["imported"] + result["labels"]["updated"]
        parts = []
        if new_labels:
            parts.append(f"{new_labels} labels imported")
        if result["model_loaded"]:
            parts.append(f"model v{imported_version} loaded")
        elif imported_version and imported_version <= local_version:
            parts.append(f"model v{imported_version} skipped (local v{local_version} is newer)")
        if not parts:
            parts.append("no new data")

        if new_labels and not result["model_loaded"]:
            parts.append("retrain recommended")

        result["message"] = ". ".join(parts).capitalize() + "."

    return result
