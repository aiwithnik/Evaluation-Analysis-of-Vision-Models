"""
Normalization and tensor-saving module.

Responsibility:
- Take RGB images in memory
- Normalize using ImageNet statistics
- Convert to tensors
- Save tensors to data/processed
"""

from pathlib import Path
import json
import torch
import numpy as np


# ImageNet normalization values (for EfficientNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def normalize_and_save(
    rgb_data,
    output_root
):
    """
    Normalize RGB images and save tensors to disk.

    Parameters
    ----------
    rgb_data : dict
        Output from rgb_converter with structure:
        {
            "train": [(img_array, label), ...],
            "val":   [(img_array, label), ...],
            "test":  [(img_array, label), ...]
        }

    output_root : str or Path
        Root directory for processed tensors (data/processed)
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Create class-to-index mapping
    all_labels = sorted(
        {label for split in rgb_data.values() for _, label in split}
    )
    class_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    # Save class mapping
    with open(output_root / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)

    # Process each split
    for split, samples in rgb_data.items():

        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_array, label) in enumerate(samples):

            # Convert to float and scale to [0, 1]
            img = img_array.astype(np.float32) / 255.0

            # Normalize (H, W, C)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

            # Convert to tensor and change shape to (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            label_tensor = torch.tensor(class_to_idx[label], dtype=torch.long)

            # Save tensors
            torch.save(
                {
                    "image": img_tensor,
                    "label": label_tensor,
                },
                split_dir / f"sample_{idx}.pt"
            )

    return True
