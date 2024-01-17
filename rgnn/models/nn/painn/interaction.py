from typing import Tuple

import torch
from torch_geometric.utils import scatter

from rgnn.models.nn.mlp import MLP
from rgnn.common.typing import Tensor


class PaiNNMixing(torch.nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: str, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.intraatomic_context_net = MLP(
            n_input=2 * n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )
        self.mu_channel_mix = torch.nn.Linear(n_atom_basis, 2 * n_atom_basis, bias=False)
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mu_channel_mix.weight.data)

    def forward(self, q: Tensor, mu: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        # intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class PaiNNInteraction(torch.nn.Module):
    r"""Copied from schnetpack.

    PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: str):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = MLP(
            n_input=n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )

    def forward(
        self, q: Tensor, mu: Tensor, Wij: Tensor, dir_ij: Tensor, idx_i: Tensor, idx_j: Tensor, n_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter(dq, idx_i, dim_size=n_atoms, dim=0)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter(dmu, idx_i, dim_size=n_atoms, dim=0)

        q = q + dq
        mu = mu + dmu

        return q, mu
