from pathlib import Path
import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """
    Dataset for loading preprocessed tensor files (.pt).
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = list(self.root_dir.glob("*.pt"))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .pt files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        return data["image"], data["label"]
