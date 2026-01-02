"""
Split standardized images into train/val/test
and save them to data/interim.

Split ratios are PROVIDED by the pipeline (from config).
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image


def split_and_save(
    images,
    labels,
    output_root,
    train_size,
    val_size,
    test_size,
    random_seed=42
):
    """
    Split images and labels, then save to disk.
    """

    output_root = Path(output_root)

    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        train_size=train_size,
        stratify=labels,
        random_state=random_seed
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_ratio,
        stratify=y_temp,
        random_state=random_seed
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split_name, (split_images, split_labels) in splits.items():
        for idx, (img_array, label) in enumerate(zip(split_images, split_labels)):

            class_dir = output_root / split_name / label
            class_dir.mkdir(parents=True, exist_ok=True)

            img = Image.fromarray(img_array)
            img.save(class_dir / f"{split_name}_{idx}.png")

    return True
