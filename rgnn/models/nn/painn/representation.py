from typing import Callable, Literal

import torch

from rgnn.common import keys as K
from rgnn.common.typing import DataDict
from rgnn.models.nn.cutoff import CosineCutoff
from rgnn.models.nn.radial_basis import BesselRBF, GaussianRBF

from .interaction import PaiNNInteraction, PaiNNMixing


def replicate_module(module_factory: Callable[[], torch.nn.Module], n: int, share_params: bool):
    if share_params:
        module_list = torch.nn.ModuleList([module_factory()] * n)
    else:
        module_list = torch.nn.ModuleList([module_factory() for i in range(n)])
    return module_list


class PaiNNRepresentation(torch.nn.Module):
    embedding_keys = (K.node_features, K.node_vec_features)

    def __init__(
        self,
        hidden_channels: int = 128,
        n_interactions: int = 3,
        rbf_type: Literal["gaussian", "bessel"] = "bessel",
        n_rbf: int = 20,
        trainable_rbf: bool = False,
        cutoff=5.0,
        activation: str = "silu",
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_interactions = n_interactions
        self.rbf_type = rbf_type
        self.n_rbf = n_rbf
        self.trainable_rbf = trainable_rbf
        self.cutoff = cutoff
        self.activation = activation
        self.shared_interactions = shared_interactions
        self.shared_filters = shared_filters
        self.epsilon = epsilon

        self.cutoff_fn = CosineCutoff(cutoff)
        if rbf_type == "gaussian":
            self.radial_basis = GaussianRBF(n_rbf, self.cutoff, trainable=trainable_rbf)
        elif rbf_type == "bessel":
            self.radial_basis = BesselRBF(n_rbf, self.cutoff, trainable=trainable_rbf)
        else:
            raise ValueError("Unknown radial basis function type: {}".format(rbf_type))

        self.embedding = torch.nn.Embedding(100, hidden_channels, padding_idx=0)

        self.share_filters = shared_filters
        if shared_filters:
            self.filter_net = torch.nn.Linear(self.radial_basis.n_rbf, 3 * hidden_channels)
        else:
            self.filter_net = torch.nn.Linear(
                self.radial_basis.n_rbf,
                self.n_interactions * hidden_channels * 3,
            )

        self.interactions = replicate_module(
            lambda: PaiNNInteraction(n_atom_basis=self.hidden_channels, activation=activation),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = replicate_module(
            lambda: PaiNNMixing(n_atom_basis=self.hidden_channels, activation=activation, epsilon=epsilon),
            self.n_interactions,
            shared_interactions,
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.filter_net.weight.data)

    def forward(self, data: DataDict) -> DataDict:
        # get tensors from input dictionary
        z = data[K.elems]
        edge_index = data[K.edge_index]  # neighbors
        edge_vec = data[K.edge_vec]
        idx_i = edge_index[1]
        idx_j = edge_index[0]

        n_atoms = z.size(0)

        # compute atom and pair features
        d_ij = torch.norm(edge_vec, dim=1, keepdim=True)
        dir_ij = edge_vec / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.hidden_channels, dim=-1)

        q = self.embedding(z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing, strict=True)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        data[K.node_features] = q  # (n_atoms, n_features)
        data[K.node_vec_features] = mu.swapaxes(-2, -1)  # (n_atoms, n_features, 3)

        return data
