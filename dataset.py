import os

from glob import glob

import numpy as np

from torch.utils.data import Dataset, Subset
from torchvision import transforms


class TrainingDataset(Dataset):
    """Simple dataset used for training."""

    def __init__(self, data_dir: str, targets_dir: str, transform: transforms.Compose | None = None):
        """Sort and store all files found in given directory."""
        self.files = sorted(os.path.abspath(f) for f in glob(os.path.join(data_dir, "**", "*"), recursive=True))
        self.targets = np.genfromtxt(targets_dir, delimiter=",", dtype=int, filling_values=-1)
        self.transform = transform

    def __getitem__(self, index: int):
        """Returns file given an index."""
        sample = self.files[index]
        target = self.targets[index]

        if self.transform is not None:
            return index, self.transform(sample), target

        return index, sample, target

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.targets)

    def split(self, train_idx, val_idx, test_idx):
        """Splits the dataset into three subsets given the indices."""
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)
