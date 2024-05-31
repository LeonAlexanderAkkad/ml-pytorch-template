from abc import ABC, abstractmethod

from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2


class TrainingDataset(Dataset, ABC):
    """Simple abstract dataset used for training."""

    def __init__(self, data_dir: str, targets_dir: str, transform: v2.Compose | None = None):
        """Sort and store all files found in given directory."""
        self.data, self.targets = self.get_data(data_dir, targets_dir)
        self.transform = transform

    def __getitem__(self, index: int):
        """Returns file given an index."""
        sample = self.load_data(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            return index, self.transform(sample), target

        return index, sample, target

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.data)

    def split(self, train_idx, val_idx, test_idx):
        """Splits the dataset into three subsets given the indices."""
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)

    @abstractmethod
    def get_data(self, data_dir: str, targets_dir: str):
        pass

    @abstractmethod
    def load_data(self, sample):
        """Method for loading the given sample such that it can be used for the transform chain."""
        pass


class TestDataset(Dataset, ABC):
    """Simple abstract dataset used for testing."""

    def __init__(self, data_dir: str):
        """Sort and store all files found in given directory."""
        self.data = self.get_data(data_dir)

    def __getitem__(self, index: int):
        """Returns file given an index."""
        sample = self.data[index]

        return index, sample

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.data)

    @abstractmethod
    def get_data(self, data_dir: str):
        pass
