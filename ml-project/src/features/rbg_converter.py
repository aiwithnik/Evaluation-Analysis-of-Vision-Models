"""
RGB conversion module.

Responsibility:
- Load grayscale images from data/interim
- Convert them to RGB (3-channel)
- Preserve split and class structure
- Return RGB images in memory (NO saving)
"""

from pathlib import Path
from PIL import Image
import numpy as np


def convert_grayscale_to_rgb(
    interim_root,
    valid_extensions=(".jpg", ".jpeg", ".png")
):
    """
    Convert grayscale images in data/interim to RGB.

    Parameters
    ----------
    interim_root : str or Path
        Root directory containing train/val/test splits
    valid_extensions : tuple
        Allowed image file extensions

    Returns
    -------
    data : dict
        Dictionary with structure:
        {
            "train": [(rgb_image_array, label), ...],
            "val":   [(rgb_image_array, label), ...],
            "test":  [(rgb_image_array, label), ...]
        }
    """

    interim_root = Path(interim_root)

    data = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Iterate over splits (train / val / test)
    for split in data.keys():

        split_dir = interim_root / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        # Iterate over class folders
        for class_dir in split_dir.iterdir():

            if not class_dir.is_dir():
                continue

            label = class_dir.name

            for img_path in class_dir.iterdir():

                if img_path.suffix.lower() not in valid_extensions:
                    continue

                # Load image
                with Image.open(img_path) as img:

                    # Ensure grayscale
                    img = img.convert("L")

                    # Convert grayscale -> RGB
                    img_rgb = img.convert("RGB")

                    # Convert to numpy array (H, W, 3)
                    img_array = np.array(img_rgb)

                data[split].append((img_array, label))

    return data
