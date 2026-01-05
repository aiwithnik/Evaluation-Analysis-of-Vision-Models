"""
Model registry with versioning.

Responsibility:
- Save models with incremental versions
- Store metadata alongside models
"""

from pathlib import Path
import json
from datetime import datetime
import torch


REGISTRY_ROOT = Path("model/registry")


def _get_next_version():
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)

    existing = [
        int(p.name.replace("v", ""))
        for p in REGISTRY_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("v")
    ]

    next_version = max(existing, default=0) + 1
    return f"v{next_version:03d}"


def save_model(
    model,
    metadata: dict,
):
    """
    Save a model with automatic versioning.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    metadata : dict
        Training metadata (metrics, config, etc.)
    """

    version = _get_next_version()
    version_dir = REGISTRY_ROOT / version
    version_dir.mkdir(parents=True)

    # Save model weights
    model_path = version_dir / "model.pt"
    torch.save(
        model.state_dict(),
        model_path,
        _use_new_zipfile_serialization=False,
    )

    # Add registry metadata
    metadata = metadata.copy()
    metadata.update(
        {
            "version": version,
            "saved_at": datetime.now().isoformat(),
        }
    )

    # Save metadata
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Update "latest" pointer
    latest_path = REGISTRY_ROOT / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(version_dir)

    return version

def load_model(model, version="latest"):
    """
    Load model weights from the registry.
    Robust to empty or broken 'latest' directories.
    """

    # Collect all valid version directories that contain model.pt
    version_dirs = sorted(
        [
            p for p in REGISTRY_ROOT.glob("v*")
            if p.is_dir() and (p / "model.pt").exists()
        ]
    )

    if not version_dirs:
        raise FileNotFoundError(
            f"No valid model.pt found in registry: {REGISTRY_ROOT}"
        )

    if version == "latest":
        # Prefer latest symlink if valid
        latest_path = REGISTRY_ROOT / "latest" / "model.pt"
        if latest_path.exists():
            model_path = latest_path
        else:
            # Fallback to highest version
            model_path = version_dirs[-1] / "model.pt"
    else:
        model_path = REGISTRY_ROOT / version / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    return model

    """
    Load model weights from the registry.
    """

    if version == "latest":
        version_dir = REGISTRY_ROOT / "latest"
    else:
        version_dir = REGISTRY_ROOT / version

    model_path = version_dir / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    return model
