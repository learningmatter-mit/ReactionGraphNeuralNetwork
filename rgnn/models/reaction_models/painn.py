from typing import Any, Dict, Literal, Tuple

import torch
from torch.nn import functional as F
from torch_geometric.utils import scatter

from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, Tensor
from rgnn.graph.utils import compute_neighbor_vecs
from rgnn.models.nn.mlp import MLP
from rgnn.models.nn.painn.representation import PaiNNRepresentation
from rgnn.models.nn.pool import AttentionPool
from rgnn.models.nn.scale import ScaleShift

from .base import BaseReactionModel


@registry.register_reaction_model("painn")
class PaiNN(BaseReactionModel):
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
        means=None,
        stddevs=None,
        pool_method="sum",
    ):
        super().__init__(species, cutoff)
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
        self.means = means
        self.stddevs = stddevs
        self.reaction_feat = reaction_feat
        self.pool_method = pool_method
        self.hyperparams = self.get_hyperparams() # TODO: Can be deleted?

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
        self.energy_output = MLP(
            n_input=hidden_channels,
            n_output=1,
            hidden_layers=(hidden_channels // 2,),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.reaction_representation = MLP(
            n_input=hidden_channels,
            n_output=reaction_feat,
            hidden_layers=(hidden_channels,),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.reaction_output = MLP(
            n_input=reaction_feat + 1,
            n_output=2,
            hidden_layers=(reaction_feat, reaction_feat),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )

        if self.pool_method == "attention":
            self.pool = AttentionPool(
                feat_dim=reaction_feat,
                att_act="silu",
                prob_func="softmax",
            )
        self.reset_parameters()
        if means is not None and stddevs is not None:
            self.scale_output = True
            self.scale_shift = ScaleShift(means=means, stddevs=stddevs)
        else:
            self.scale_output = False

    def reset_parameters(self):
        self.energy_output.reset_parameters()
        self.representation.reset_parameters()
        self.reaction_representation.reset_parameters()
        self.reaction_output.reset_parameters()

    def forward(self, data: DataDict, get_features: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        compute_neighbor_vecs(data)
        data = self.representation(data)
        mask_tensor_r = self.create_subgraph_mask(data)
        # Compute per-atom energy
        energy_combined = self.energy_output(data[K.node_features]).squeeze(-1)
        energy_combined = self.species_energy_scale(data, energy_combined)
        energy_r = energy_combined[mask_tensor_r]
        energy_p = energy_combined[~mask_tensor_r]

        # Compute system energy
        energy_total_r = scatter(energy_r, data[K.batch][mask_tensor_r], dim=0, reduce="sum")
        energy_total_p = scatter(energy_p, data[K.batch][~mask_tensor_r], dim=0, reduce="sum")
        energy_reaction = torch.sub(energy_total_p, energy_total_r)

        # Reaction
        node_feat_r = data[K.node_features][mask_tensor_r]
        node_feat_f = data[K.node_features][~mask_tensor_r]
        feature_diff = torch.sub(node_feat_f, node_feat_r)
        reaction_feat = self.reaction_representation(feature_diff)
        if self.pool_method == "attention":
            reaction_feat = self.pool(data, reaction_feat)
        elif self.pool_method == "sum":
            reaction_feat = scatter(reaction_feat, data[K.batch][mask_tensor_r], dim=0, reduce="sum")
        reaction_feat = torch.cat([energy_reaction.unsqueeze(-1), reaction_feat], dim=-1)
        reaction_feat = F.normalize(reaction_feat, dim=-1)
        data[K.reaction_features] = reaction_feat

        barrier_out = self.reaction_output(reaction_feat)
        barrier, freq = torch.chunk(barrier_out, chunks=2, dim=1)
        if self.scale_output:
            barrier = self.scale_shift(K.barrier, barrier)
            freq = self.scale_shift(K.freq, freq)

        outputs = {}
        delta_e = energy_total_p - energy_total_r
        if self.scale_output:
            delta_e = self.scale_shift(K.delta_e, delta_e)
        if get_features:
            outputs[K.reaction_features] = reaction_feat
        outputs[K.energy_i] = energy_total_r
        outputs[K.energy_f] = energy_total_p
        outputs[K.delta_e] = delta_e
        outputs[K.barrier] = barrier.squeeze(-1)
        outputs[K.freq] = freq.squeeze(-1)

        return outputs

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "name": "painn",
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
            "pool_method": self.pool_method
        }
        return hyperparams

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.get_config()
        state = {"state_dict": state_dict, "hyper_parameters": hyperparams}

        torch.save(state, filename)

    # @torch.jit.unused
    # @classmethod
    # def load(cls, path: str):
    #     """Load the model from checkpoint created by pytorch lightning.

    #     Args:
    #         path (str): Path to the checkpoint file.

    #     Returns:
    #         InterAtomicPotential: The loaded model.
    #     """
    #     map_location = None if torch.cuda.is_available() else "cpu"
    #     ckpt = torch.load(path, map_location=map_location)
    #     hparams = ckpt["hyper_parameters"]
    #     del hparams["name"]
    #     state_dict = ckpt["state_dict"]
    #     model = cls(**hparams)
    #     print(state_dict.keys())
    #     print(model.pool_method)
    #     model.load_state_dict(state_dict=state_dict)
    #     return model
