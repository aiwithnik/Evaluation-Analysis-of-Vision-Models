import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import re


def _get_next_experiment_id(experiments_root: Path) -> str:
    experiments_root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^exp(\d{3})$")

    existing_ids = []

    for p in experiments_root.iterdir():
        if not p.is_dir():
            continue

        match = pattern.match(p.name)
        if match:
            existing_ids.append(int(match.group(1)))

    next_id = max(existing_ids, default=0) + 1
    return f"exp{next_id:03d}"


def create_experiment_dir(
    experiments_root: Path,
    *,
    experiment_name: str,
    family: str,
    scenario: str,
    model_version: str,
    degradations: list[dict],
    execution: dict,
    description: str = "",
) -> Path:
    """
    Create a new experiment directory and write metadata.

    Returns
    -------
    Path
        Path to the experiment directory.
    """
    exp_id = _get_next_experiment_id(experiments_root)
    exp_dir = experiments_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "family": family,
        "scenario": scenario,
        "description": description,
        "model": {
            "version": model_version,
        },
        "degradations": degradations,
        "execution": execution,
        "timestamps": {
            "created_at": datetime.now().isoformat(),
        },
    }

    with open(exp_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    return exp_dir


def finalize_experiment(
    exp_dir: Path,
    *,
    notes: str | None = None,
):
    """
    Finalize experiment metadata after execution.
    """
    meta_path = exp_dir / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)

    meta["timestamps"]["finished_at"] = datetime.now().isoformat()

    if notes:
        meta["notes"] = notes

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
