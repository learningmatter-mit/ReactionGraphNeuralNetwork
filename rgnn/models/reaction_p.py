from abc import ABC
from typing import Any, Dict, Literal, Tuple

import torch
from torch.nn import functional as F
from torch_geometric.utils import scatter
from torch_geometric.utils import degree
from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, Tensor
from rgnn.graph.utils import compute_neighbor_vecs
from rgnn.models.nn.mlp import MLP
from rgnn.models.nn.painn.representation import PaiNNRepresentation
from rgnn.models.nn.scale import PerSpeciesScaleShift, canocialize_species


class PNet(torch.nn.Module, ABC):
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

    embedding_keys = [K.node_features, K.reaction_features]
    # embedding_keys = [K.node_features, K.node_vec_features, K.reaction_features]

    def __init__(
        self,
        species,
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        reaction_feat: int = 32,
        n_interactions: int = 3,
        rbf_type: Literal["gaussian", "bessel"] = "bessel",
        n_rbf: int = 20,
        trainable_rbf: bool = False,
        activation: str = "silu",
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
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
        self.reaction_feat = reaction_feat
        self.hyperparams = self.get_hyperparams()

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
        self.probs_out = MLP(
            n_input=hidden_channels + 1,
            n_output=1,
            hidden_layers=(hidden_channels // 2, hidden_channels // 2),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.representation.reset_parameters()
        self.probs_out.reset_parameters()

    def forward(self, data: DataDict, kT: float):
        compute_neighbor_vecs(data)
        data = self.representation(data)

        node_degrees = degree(data["edge_index"][0], num_nodes=torch.sum(data["n_atoms"]))
        new_features_list = []
        for graph_id in torch.unique(data["batch"]):
            mask = data["batch"] == graph_id
            filtered_nodes = node_degrees[mask]

            _, sorted_index = torch.sort(filtered_nodes[mask], dim=-1, stable=True)
            connectivity_feat = sorted_index / data["n_atoms_i"][graph_id]

            new_features_list.append(connectivity_feat.unsqueeze(-1))
        connectivity_feat = torch.concat(new_features_list, dim=0)
        softmaxed_probabilities_list = []
        probs_feat = torch.concat([connectivity_feat, data[K.node_features]], dim=-1)
        probability = self.probs_out(probs_feat).squeeze(-1)
        sorted_batch, _ = torch.sort(torch.unique(data["batch"]))
        for graph_id in sorted_batch:
            mask = data["batch"] == graph_id
            graph_probs = probability[mask]
            softmaxed_probs = F.softmax(graph_probs / kT, dim=-1)
            softmaxed_probabilities_list.append(softmaxed_probs.unsqueeze(-1))
        # softmaxed_probabilities = torch.stack(softmaxed_probabilities_list, dim=0)

        return softmaxed_probabilities_list

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "name": "painn_reaction",
            "species": self.species,
            "cutoff": self.cutoff,
            "hidden_channels": self.hidden_channels,
            "reaction_feat": self.reaction_feat,
            "n_interactions": self.n_interactions,
            "rbf_type": self.rbf_type,
            "n_rbf": self.n_rbf,
            "trainable_rbf": self.trainable_rbf,
            "activation": self.activation,
            "shared_interactions": self.shared_interactions,
            "shared_filters": self.shared_filters,
            "means": self.means,
            "stddevs": self.stddevs,
        }
        return hyperparams

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.hyperparams
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
        del hparams["name"]
        state_dict = ckpt["state_dict"]
        model = cls(**hparams)
        model.load_state_dict(state_dict=state_dict)
        return model
