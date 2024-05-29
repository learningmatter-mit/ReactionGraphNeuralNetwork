from abc import ABC
from typing import Any, Dict, List, Literal, Tuple

import torch
from torch.nn import functional as F
from torch_geometric.utils import degree, scatter

from rgnn.common import keys as K
from rgnn.common.configuration import Configurable
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, Tensor
from rgnn.graph.utils import compute_neighbor_vecs
from rgnn.models.nn.mlp import MLP
from rgnn.models.nn.painn.representation import PaiNNRepresentation
from rgnn.models.nn.scale import PerSpeciesScaleShift, canocialize_species


@registry.register_model("t_net")
class TNet(torch.nn.Module, Configurable, ABC):
    """PaiNN model, as described in https://arxiv.org/abs/2102.03150.
    This model applies equivariant message passing layers within cartesian coordinates.
    Provides "node_features" and "node_vec_features" embeddings.

    Args:
        species (list[str]): List of atomic species to consider.
        cutoff (float): Cutoff radius for interactions. Defaults to 5.0.
        hidden_channels (int): Number of hidden channels in the convolutional layers. Defaults to 128.
        n_interactions (int): Number of message passing layers. Defaults to 3.
        rbf_type (str): Type of radial basis functions. One of "gaussian" or "bessel".
            Defaults to "bessel".
        n_rbf (int): Number of radial basis functions. Defaults to 20.
        trainable_rbf (bool): Whether to train the radial basis functions. Defaults to False.
        activation (str): Activation function to use in the convolutional layers. Defaults to "silu".
        shared_interactions (bool): Whether to share the convolutional layers across interactions.
            Defaults to False.
        shared_filters (bool): Whether to share the convolutional filters across interactions.
            Defaults to False.
        epsilon (float): Small value to add to the denominator for numerical stability. Defaults to 1e-8.
    """

    embedding_keys = [K.node_features]
    # embedding_keys = [K.node_features, K.node_vec_features, K.reaction_features]

    def __init__(
        self,
        species,
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        n_interactions: int = 3,
        rbf_type: Literal["gaussian", "bessel"] = "bessel",
        n_rbf: int = 20,
        trainable_rbf: bool = False,
        activation: str = "silu",
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        # canonical: bool = False,
        # N_emb: int = 16,
        n_feat: int = 32,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        atomic_numbers = [val.item() for val in canocialize_species(species)]
        atomic_numbers_dict = {}
        for i, key in enumerate(atomic_numbers):
            atomic_numbers_dict.update({key: species[i]})
        atomic_numbers.sort()
        self.atomic_numbers = atomic_numbers
        sorted_species = []
        for n in atomic_numbers:
            sorted_species.append(atomic_numbers_dict[n])
        self.species = sorted_species
        self.cutoff = cutoff
        self.species_energy_scale = PerSpeciesScaleShift(species)
        self.embedding_keys = self.__class__.embedding_keys
        # TODO: Should be more rigorous
        self.hidden_channels = hidden_channels
        self.n_interactions = n_interactions
        self.rbf_type = rbf_type
        self.n_rbf = n_rbf
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.shared_interactions = shared_interactions
        self.shared_filters = shared_filters
        self.epsilon = epsilon
        # self.canonical = self.canonical
        self.dropout_rate = dropout_rate
        self.n_feat = n_feat

        self.representation = PaiNNRepresentation(
            hidden_channels=hidden_channels,
            n_interactions=n_interactions,
            rbf_type=rbf_type,
            n_rbf=n_rbf,
            trainable_rbf=trainable_rbf,
            cutoff=cutoff,
            activation=activation,
            shared_interactions=shared_interactions,
            shared_filters=shared_filters,
            epsilon=epsilon,
        )
        self.time_out = MLP(
            n_input=hidden_channels + 1,
            n_output=1,
            hidden_layers=(self.n_feat, self.n_feat),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.kb = 8.617333262145e-5
        self.reset_parameters()

    def reset_parameters(self):
        self.representation.reset_parameters()
        self.time_out.reset_parameters()

    def forward(self, data: DataDict, temperature: float | None = None):
        """_summary_

        Args:
            data (DataDict): _description_
            temperature (float, optional): _description_. Defaults to .

        Returns:
            _type_: _description_
        """
        if "T" not in data.keys():
            kT = torch.tensor(temperature * self.kb, device=data[K.node_features].device, dtype=data[K.node_features].dtype, requires_grad=False)
        else:
            kT = torch.tensor(data["T"] * self.kb, requires_grad=False).unsqueeze(-1)
        compute_neighbor_vecs(data)
        data = self.representation(data)
        atomwise_out = self.time_out(
            torch.cat([kT, data[K.node_features]], dim=-1)
        ).squeeze(-1)
        time = scatter(atomwise_out, data[K.batch], dim=0, reduce="sum")

        return time

    def get_config(self):
        config = {}
        config["@name"] = self.__class__.name
        config.update(super().get_config())
        return config
    
    @classmethod
    def load_representation(cls, reaction_model, n_feat, dropout_rate):
        # Extract configuration from the existing model
        reaction_model_params = reaction_model.get_config()
        
        # Define the keys that you want to copy from the existing model's config
        input_keys = [
            "species", "cutoff", "hidden_channels", "n_interactions",
            "rbf_type", "n_rbf", "trainable_rbf", "activation",
            "shared_interactions", "shared_filters", "epsilon"
        ]
        
        # Create a dictionary of parameters to initialize the new model
        input_params = {key: reaction_model_params[key] for key in input_keys}
        
        # Initialize the new model with the copied parameters and additional features
        model = cls(**input_params, n_feat=n_feat, dropout_rate=dropout_rate)
        
        # Load only the state_dict entries that include 'representation' in their key,
        # assuming these belong to the shared blocks or relevant components
        loaded_state_dict = reaction_model.state_dict()
        shared_block_keys = {k: v for k, v in loaded_state_dict.items() if 'representation' in k}
        
        # Partially load state_dict into the new model, without requiring an exact match
        model.load_state_dict(shared_block_keys, strict=False)
        
        return model
        
    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.get_config()
        state = {"state_dict": state_dict, "hyper_parameters": hyperparams}

        torch.save(state, filename)

    # "best_metric": best_metric,
    @torch.jit.unused
    @classmethod
    def load(cls, path: str):
        """Load the model from checkpoint created by pytorch lightning.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            InterAtomicPotential: The loaded model.
        """
        map_location = None if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=map_location)
        hparams = ckpt["hyper_parameters"]
        hparams.pop("@name")
        state_dict = ckpt["state_dict"]
        model = cls.from_config(hparams)
        model.load_state_dict(state_dict=state_dict)
        return model
