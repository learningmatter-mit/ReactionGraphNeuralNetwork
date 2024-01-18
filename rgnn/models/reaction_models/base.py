from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn

from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, Tensor
from rgnn.models.nn.scale import PerSpeciesScaleShift


@registry.register_reaction_model("base")
class BaseReactionModel(torch.nn.Module, ABC):
    """Base class for energy models.
    Takes data or batch and returns energy.

    Args:
        species (list[str]): List of species.
        cutoff (float, optional): Cutoff radius. Defaults to 5.0.

    Attributes:
        embedding_keys (list[str]): List of keys for storing embedding vectors.
    """

    embedding_keys: List[str] = []

    def __init__(self,
                 species: list[str],
                 cutoff: float = 5.0,
                 *args,
                 **kwargs):
        super().__init__()
        self.species = species
        self.cutoff = cutoff
        self.species_energy_scale = PerSpeciesScaleShift(species)
        self.embedding_keys = self.__class__.embedding_keys

    @abstractmethod
    def forward(self, data: DataDict):
        """Compute energy from data.

        Args:
            data (DataDict): Input data.
                Required keys are:
                    - "pos"
                    - "elems"
                    - "cell"
                    - "edge_index"
                    - "edge_shift"
        Returns:
            Tensor: (N,) tensor of energies. (N: number of structures)
        """
        pass

    @torch.jit.ignore
    def get_cutoff(self) -> float:
        """Get the cutoff radius of the model."""
        return self.cutoff

    def get_config(self):
        config = {}
        config["@name"] = self.__class__.name
        config.update(super().get_config())
        return config

    @classmethod
    def from_config(cls, config: dict):
        config = deepcopy(config)
        name = config.pop("@name", None)
        if cls.__name__ == "BaseReactionModel":
            if name is None:
                raise ValueError(
                    "Cannot initialize BaseEnergyModel from config. Please specify the name of the model."
                )
            model_class = registry.get_reaction_model_class(name)
        else:
            if name is not None and hasattr(cls, "name") and cls.name != name:
                raise ValueError(
                    "The name in the config is different from the class name.")
            model_class = cls
        return super().from_config(config, actual_cls=model_class)

    @torch.jit.unused
    @property
    def num_params(self):
        """Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def create_subgraph_mask(self, data):
        """
        Create a mask tensor for the initial subgraphs in each data point of the batch.

        :param batch: Tensor indicating the data point each node belongs to.
        :param n_atoms_i: Tensor with the number of nodes in the initial subgraph of each data point.
        :param n_atoms_f: Tensor with the number of nodes in the final subgraph of each data point.
        :return: Mask tensor indicating nodes belonging to the initial subgraphs.
        """
        # Initialize mask tensor of the same size as batch, filled with False
        mask_tensor = torch.zeros_like(data[K.batch],
                                       dtype=torch.bool,
                                       device=data[K.batch].device)

        # Iterate through each data point in the data[K.batch]
        for data_point in torch.unique(data[K.batch]):
            # Find the indices where this data point appears in the data[K.batch]
            indices = (data[K.batch] == data_point).nonzero(as_tuple=True)[0]

            # Calculate the start and end index for the initial subgraph of this data point
            start_idx = indices[0]
            end_idx = start_idx + data[K.n_atoms_i][data_point]

            # Set the corresponding elements in the mask tensor to True
            mask_tensor[start_idx:end_idx] = 1

        return mask_tensor
