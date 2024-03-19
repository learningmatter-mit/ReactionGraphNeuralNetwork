from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn

from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.configuration import Configurable
from rgnn.common.typing import DataDict


@registry.register_model("base")
class BaseDQN(torch.nn.Module, Configurable, ABC):
    """Base class for energy models.
    Takes data or batch and returns energy.

    Args:
        species (list[str]): List of species.
        cutoff (float, optional): Cutoff radius. Defaults to 5.0.

    Attributes:
        embedding_keys (list[str]): List of keys for storing embedding vectors.
    """

    embedding_keys: List[str] = []

    def __init__(self, reaction_model, *args, **kwargs):
        super().__init__()
        self.reaction_model = reaction_model
        self.cutoff = self.reaction_model.get_cutoff()
        self.kb = 8.617 * 10**-5

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

    @torch.jit.unused
    @property
    def num_params(self):
        """Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    @torch.jit.unused
    @property
    def output_keys(self) -> tuple[str, ...]:
        keys = [K.delta_e, K.barrier, K.freq, K.rl_q]
        return tuple(keys)

    @torch.jit.unused
    def get_change_n_elems(self, data):
        """To Compute the change in number of elements between reactants and products

        Args:
            data (_type_): _description_

        Returns:
            change_elems (Tensor): Change in number of elements excluding "X"
        """
        mask_tensor_r = self.reaction_model.create_subgraph_mask(data)
        change_elem_list = [{} for _ in range(torch.unique(data["batch"]).shape[0])]
        for _, graph_id in enumerate(torch.unique(data["batch"])):
            mask = data["batch"] == graph_id
            reactant_elems = data["elems"][mask][mask_tensor_r[mask]]
            product_elems = data["elems"][mask][~mask_tensor_r[mask]]
            elem_change_dict = {}
            unique2, counts2 = torch.unique(product_elems, return_counts=True)
            unique1, counts1 = torch.unique(reactant_elems, return_counts=True)
            change_el = torch.zeros(
                len(self.reaction_model.atomic_numbers), dtype=torch.float, device=data["batch"].device
            )
            for i, specie in enumerate(unique2):
                elem_change_dict[specie.item()] = counts2[i]
            for i, specie in enumerate(unique1):
                count = elem_change_dict.get(
                    specie.item(), torch.tensor(0.0, dtype=torch.float, device=data["batch"].device)
                )
                new_count = count - counts1[i]
                elem_change_dict.update({specie.item(): new_count})
            for i, el in enumerate(self.reaction_model.atomic_numbers):
                change_el[i] = elem_change_dict[el]

            change_elem_list[graph_id] = change_el[1:]
        change_elems = torch.stack(change_elem_list, dim=0)
        return change_elems

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.get_config()
        state = {
            "state_dict": state_dict,
            "hyper_parameters": hyperparams,
        }

        torch.save(state, filename)

    @classmethod
    def load(cls, path: str) -> "BaseDQN":
        """Load the model from checkpoint created by pytorch lightning.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            InterAtomicPotential: The loaded model.
        """
        map_location = None if torch.cuda.is_available() else "cpu"
        # if str(path).endswith(".ckpt"):
        ckpt = torch.load(path, map_location=map_location)
        hparams = ckpt["hyper_parameters"]
        hparams.pop("@name")
        reaction_model_hparams = hparams.pop("reaction_model")
        reaction_model_name = reaction_model_hparams.pop("@name")

        state_dict = ckpt["state_dict"]
        reaction_model = registry.get_reaction_model_class(reaction_model_name)(**reaction_model_hparams)
        model = cls(reaction_model, **hparams)
        model.load_state_dict(state_dict=state_dict)
        return model

    @classmethod
    def load_old(cls, path: str) -> "BaseDQN":
        """Load the model from checkpoint created by pytorch lightning.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            InterAtomicPotential: The loaded model.
        """
        map_location = None if torch.cuda.is_available() else "cpu"
        # if str(path).endswith(".ckpt"):
        ckpt = torch.load(path, map_location=map_location)
        hparams = ckpt["hyper_parameters"]
        if hparams.get("name", None) is not None:
            reaction_model_name = hparams["name"]
            hparams.pop("name")
        elif hparams.get("@name", None) is not None:
            reaction_model_name = hparams["@name"]
            hparams.pop("@name")

        # model_config = hparams["hyperparams"]
        state_dict = ckpt["state_dict"]
        N_feat = ckpt.get("n_feat", 32)  # TODO: Should be changed
        N_emb = ckpt.get("n_emb", 16)  # TODO: Should be changed
        dropout_rate = ckpt.get("dropout_rate", 0.15)
        canonical = ckpt.get("canonical", False)
        # state_dict = {
        #     ".".join(k.split(".")[1:]): v for k, v in state_dict.items()
        # }
        if reaction_model_name == "painn_reaction2":
            reaction_model_name = "painn"
        reaction_model = registry.get_reaction_model_class(reaction_model_name)(**hparams)
        # reaction_model = get_model(hparams)
        model = cls(reaction_model, N_emb, N_feat, dropout_rate, canonical)
        model.load_state_dict(state_dict=state_dict)
        return model
