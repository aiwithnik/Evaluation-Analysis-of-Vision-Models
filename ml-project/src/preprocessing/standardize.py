"""
Image standardization (in-memory).

Responsibility:
- Load images
- Convert to grayscale
- Resize to a fixed size
- Return standardized images and labels (no saving)
"""

from pathlib import Path
from PIL import Image
import numpy as np


def standardize_images(
    dataset_path,
    image_size=(224, 224),
    valid_extensions=(".jpg", ".jpeg", ".png")
):
    """
    Load, convert to grayscale, and resize images.

    Parameters
    ----------
    dataset_path : str or Path
        Root directory with class-wise folders
    image_size : tuple
        Target image size (width, height)
    valid_extensions : tuple
        Allowed image file extensions

    Returns
    -------
    images : list[np.ndarray]
        Standardized images as numpy arrays
    labels : list[str]
        Corresponding class labels
    """

    dataset_path = Path(dataset_path)

    images = []
    labels = []

    # Iterate over class folders
    for class_dir in dataset_path.iterdir():

        if not class_dir.is_dir():
            continue

        class_label = class_dir.name

        for img_path in class_dir.iterdir():

            if img_path.suffix.lower() not in valid_extensions:
                continue

            # Load image
            with Image.open(img_path) as img:
                # Convert to grayscale
                img = img.convert("L")

                # Resize image
                img = img.resize(image_size)

                # Convert to numpy array
                img_array = np.array(img)

            images.append(img_array)
            labels.append(class_label)

    return images, labels
