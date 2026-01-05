import json
from pathlib import Path
from datetime import datetime

from src.config import load_yaml, PATHS_YAML


def _get_next_report_id(reports_root: Path) -> str:
    """
    Generate the next report ID (r001, r002, ...).
    """
    reports_root.mkdir(parents=True, exist_ok=True)

    existing = [
        int(p.name.replace("r", ""))
        for p in reports_root.iterdir()
        if p.is_dir() and p.name.startswith("r")
    ]

    next_id = max(existing, default=0) + 1
    return f"r{next_id:03d}"


def create_report_dir() -> Path:
    """
    Create and return a new report directory.
    """
    paths = load_yaml(PATHS_YAML)
    reports_root = Path(paths["reports"]["root"])

    report_id = _get_next_report_id(reports_root)
    report_dir = reports_root / report_id
    report_dir.mkdir(parents=True, exist_ok=False)

    return report_dir


def save_metrics(report_dir: Path, metrics: dict):
    """
    Save evaluation metrics as JSON.
    """
    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_report_metadata(
    report_dir: Path,
    *,
    model_version: str,
    evaluation_type: str,
    params: dict | None = None,
):
    """
    Save report metadata describing the evaluation context.
    """
    metadata = {
        "report_id": report_dir.name,
        "model_version": model_version,
        "evaluation_type": evaluation_type,
        "created_at": datetime.now().isoformat(),
    }

    if params:
        metadata.update(params)

    with open(report_dir / "report_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)
