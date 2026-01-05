from pathlib import Path
from src.config import load_yaml, PATHS_YAML


def get_interim_split_path(split: str) -> Path:
    """
    Returns the path to a specific split inside the interim dataset.

    Expected structure:
        data/interim/
            ├── train/
            ├── val/
            └── test/
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: train, val, test"
        )

    paths = load_yaml(PATHS_YAML)
    interim_root = Path(paths["data"]["interim"])

    if not interim_root.exists():
        raise FileNotFoundError(
            f"Interim data root not found at: {interim_root}"
        )

    split_path = interim_root / split

    if not split_path.exists():
        raise FileNotFoundError(
            f"Interim split not found at: {split_path}"
        )

    if not split_path.is_dir():
        raise NotADirectoryError(
            f"Expected directory for split '{split}', got file: {split_path}"
        )

    return split_path
