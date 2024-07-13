from typing import Dict, List, Union

import torch
from torch import nn

from rgnn.common.typing import Tensor
from rgnn.graph.dataset.reaction import ReactionDataset


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(
        self,
        inputs: Union[Tensor, Dict],
    ):
        """tensor is taken as a sample to calculate the mean and std"""
        if isinstance(inputs, Tensor):
            self.mean, self.std, self.max, self.min, self.sum = self.preprocess(inputs)

        elif isinstance(inputs, Dict):
            self.load_state_dict(inputs)

        else:
            TypeError

    def norm(self, tensor):
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        return (tensor - mean) / std
        # return (tensor - self.mean) / self.std

    def norm_to_unity(self, tensor):
        _sum = self.sum.to(tensor.device)

        return tensor / _sum

    def denorm(self, normed_tensor):
        std = self.std.to(normed_tensor.device)
        mean = self.mean.to(normed_tensor.device)

        return normed_tensor * std + mean
        # return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "max": self.max,
            "min": self.min,
            "sum": self.sum,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.max = state_dict["max"]
        self.min = state_dict["min"]
        self.sum = state_dict["sum"]

    def preprocess(
        self,
        tensor,
    ):
        """
        Preprocess the tensor:
        (1) filter nan
        (2) calculate mean, std, max, min, sum depending on the dimension
        """
        if tensor.dim() == 1:
            valid_index = torch.bitwise_not(torch.isnan(tensor))
            filtered_targs = tensor[valid_index]
            mean = torch.mean(filtered_targs)
            std = torch.std(filtered_targs)
            _max = torch.max(filtered_targs)
            _min = torch.min(filtered_targs)
            _sum = torch.sum(filtered_targs)
        elif tensor.dim() == 2:
            mean_bin = []
            std_bin = []
            _max_bin = []
            _min_bin = []
            _sum_bin = []
            transposed = torch.transpose(tensor, dim0=0, dim1=1)
            for i, values in enumerate(transposed):
                valid_index = torch.bitwise_not(torch.isnan(values))
                filtered_targs = values[valid_index]
                mean_temp = torch.mean(filtered_targs)
                mean_bin.append(mean_temp)
                std_temp = torch.std(filtered_targs)
                std_bin.append(std_temp)
                max_temp = torch.max(filtered_targs)
                _max_bin.append(max_temp)
                min_temp = torch.min(filtered_targs)
                _min_bin.append(min_temp)
                sum_temp = torch.sum(filtered_targs)
                _sum_bin.append(sum_temp)

            mean = torch.tensor(mean_bin)
            std = torch.tensor(std_bin)
            _max = torch.tensor(_max_bin)
            _min = torch.tensor(_min_bin)
            _sum = torch.tensor(_sum_bin)
        else:
            ValueError("input dimension is not right.")

        return mean, std, _max, _min, _sum


def get_scaler(keys: List[str], dataset: ReactionDataset):
    means = {}
    stddevs = {}
    for K in keys:
        values = torch.tensor([data[K] for data in dataset])
        normalizer = Normalizer(values)
        means.update({K: normalizer.mean})
        stddevs.update({K: normalizer.std})
    return means, stddevs
