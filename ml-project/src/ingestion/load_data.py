"""
Locate the raw dataset inside the project directory.

This module assumes the dataset is already downloaded manually.
"""

from pathlib import Path
from src.config import load_yaml, PATHS_YAML


def get_raw_data_path():
    paths = load_yaml(PATHS_YAML)

    raw_root = Path(paths["data"]["raw"])
    dataset_name = paths["data"]["raw_dataset_name"]

    dataset_path = raw_root / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {dataset_path}\n"
            "Please place the dataset inside data/raw/"
        )

    return dataset_path
