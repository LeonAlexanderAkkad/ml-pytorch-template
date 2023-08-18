from os import path
from glob import glob
from torch.utils.data import Dataset, Subset


class TrainingDataset(Dataset):
    """Simple dataset used for training."""

    def __init__(self, directory: str):
        """Sort and stor all files found in given directory."""
        self.files = sorted(path.abspath(f) for f in glob(path.join(directory, "**", "*"), recursive=True))

    def __getitem__(self, index: int):
        """Returns file given an index."""
        return self.files[index]

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.files)

    def split(self, train_idx, val_idx, test_idx):
        """Splits the dataset into three subsets given the indices."""
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)
