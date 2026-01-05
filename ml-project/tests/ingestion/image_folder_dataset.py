from pathlib import Path
from PIL import Image
import torch
import numpy as np


class ImageFolderDataset:
    """
    Loads images from a folder structured as:
    root/
      class_a/
      class_b/
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.uint8)

        return img, label
