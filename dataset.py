import os

from glob import glob

from torch.utils.data import Dataset, Subset
from torchvision import transforms


class TrainingDataset(Dataset):
    """Simple dataset used for training."""

    def __init__(self, path: str, transform: transforms.Compose | None = None):
        """Sort and store all files found in given directory."""
        self.files = sorted(os.path.abspath(f) for f in glob(os.path.join(path, "**", "*"), recursive=True))
        self.transform = transform

    def __getitem__(self, index: int):
        """Returns file given an index."""
        sample = self.files[index]

        if self.transform:
            return self.transform(sample), index

        return sample, index

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.files)

    def split(self, train_idx, val_idx, test_idx):
        """Splits the dataset into three subsets given the indices."""
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)
