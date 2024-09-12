import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class AtomsDataset(TorchDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # If idx is a tensor, convert it to a list or numpy array of integers
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()  # Convert the tensor to a list of integers
        
        # Handle the case where idx is a list or array of indices
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            return [self.data_list[i] for i in idx]  # Return a list of the corresponding items

        # Handle the case where idx is a single integer
        return self.data_list[idx]

    def split(self, size: int | float, seed: int = 0, return_idx: bool = False):
        """Split the dataset into two subsets of `size` and `len(self) - size` elements.

        Args:
            size (int | float): Size of the first subset. If float, it is interpreted as the fraction of the dataset.
            seed (int, optional): Random seed. Defaults to 0.
            return_idx (bool, optional): Whether to return indices of the subsets. Defaults to False.

        Returns:
            Tuple[Self, Self]: Two subsets of the dataset.
        """
        size = _determine_size(self, size)
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return (self[indices[:size]], self[indices[size:]]), (indices[:size], indices[size:])
        return self[indices[:size]], self[indices[size:]]

    def train_val_test_split(
        self,
        train_size: int | float,
        val_size: int | float,
        seed: int = 0,
        return_idx: bool = False,
    ):
        """Split the dataset into three subsets of `train_size`, `val_size`, and `len(self) - train_size - val_size`
        elements.

        Args:
            train_size (int | float): Size of the training subset. If float, it is interpreted as the fraction of the
                dataset.
            val_size (int | float): Size of the validation subset. If float, it is interpreted as the fraction of the
                dataset.
            seed (int, optional): Random seed. Defaults to 0.
            return_idx (bool, optional): Whether to return indices of the subsets. Defaults to False.

        Returns:
            Tuple[Self, Self, Self]: Three subsets of the dataset.
        """
        train_size = _determine_size(self, train_size)
        val_size = _determine_size(self, val_size)

        if return_idx:
            (train_dataset, rest_dataset), (train_idx, _) = self.split(train_size, seed, return_idx)
            (val_dataset, test_dataset), (val_idx, test_idx) = rest_dataset.split(val_size, seed, return_idx)
            return (train_dataset, val_dataset, test_dataset), (train_idx, val_idx, test_idx)

        train_dataset, rest_dataset = self.split(train_size, seed)
        val_dataset, test_dataset = rest_dataset.split(val_size, seed)
        return train_dataset, val_dataset, test_dataset
    
    def train_val_split(
        self,
        train_size: int | float,
        seed: int = 0,
        return_idx: bool = False,
    ):
        """Split the dataset into three subsets of `train_size`, `val_size`, and `len(self) - train_size - val_size`
        elements.

        Args:
            train_size (int | float): Size of the training subset. If float, it is interpreted as the fraction of the
                dataset.
            val_size (int | float): Size of the validation subset. If float, it is interpreted as the fraction of the
                dataset.
            seed (int, optional): Random seed. Defaults to 0.
            return_idx (bool, optional): Whether to return indices of the subsets. Defaults to False.

        Returns:
            Tuple[Self, Self, Self]: Three subsets of the dataset.
        """
        train_size = _determine_size(self, train_size)

        if return_idx:
            (train_dataset, val_dataset), (train_idx, val_idx) = self.split(train_size, seed, return_idx)
            return (train_dataset, val_dataset), (train_idx, val_idx)

        train_dataset, val_dataset = self.split(train_size, seed)
        return train_dataset, val_dataset


def _determine_size(dataset: TorchDataset, size: int | float) -> int:
    if isinstance(size, float):
        size = int(len(dataset) * size)
    elif isinstance(size, int):
        size = size
    else:
        raise TypeError(f"size must be int or float, not {type(size)}")
    if size > len(dataset):
        raise ValueError(f"size must be less than or equal to the length of dataset, {len(dataset)}, but got {size}")
    return size
