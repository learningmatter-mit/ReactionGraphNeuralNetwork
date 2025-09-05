from typing import Dict

import torch
from torch.nn import functional as F

from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, OutputDict, Tensor
from rgnn.models.nn.mlp import MLP
from rgnn.models.reaction_models.base import BaseReactionModel

from .base import BaseDQN


@registry.register_model("dqn_v2")
class ReactionDQN2(BaseDQN):
    """DQN model based on the reaction model.

    Args:
        reaction_model (BaseReactionModel): The reaction model.
        N_emb (int, optional): The embedding dimension. Defaults to 16.
        N_feat (int, optional): The hidden dimension. Defaults to 32.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.15.
        canonical (bool, optional): Whether to use the canonical model. Defaults to False.
    """

    def __init__(
        self,
        reaction_model: BaseReactionModel,
        N_emb: int = 16,
        N_feat: int = 32,
        dropout_rate: float = 0.15,
        canonical: bool = False,
        mean_mu: Dict[str, float] | None = None,
        max_mu: Dict[str, float] | None = None,
    ):
        super().__init__(reaction_model)
        self.N_feat = N_feat
        self.N_emb = N_emb
        self.dropout_rate = dropout_rate
        self.canonical = canonical
        self.mean_mu = mean_mu
        self.max_mu = max_mu
        n_input = N_emb + 2 if canonical else N_emb + len(reaction_model.atomic_numbers) + 1
        self.Q_emb = MLP(
            n_input=reaction_model.reaction_feat + 1,
            n_output=N_emb,
            hidden_layers=(N_emb,),
            activation="leaky_relu",
            w_init="xavier_uniform",
            b_init="zeros",
            dropout_rate=self.dropout_rate,
        )
        self.Q_output = MLP(
            n_input=n_input,
            n_output=1,
            hidden_layers=(N_feat, N_feat),
            activation="leaky_relu",
            w_init="xavier_uniform",
            b_init="zeros",
            dropout_rate=self.dropout_rate,
        )
        self.Q_emb.reset_parameters()
        self.Q_output.reset_parameters()

    def set_chempot_stats(self, chempot_stats):
        self.mean_mu = chempot_stats["mean_mu"]
        self.max_mu = chempot_stats["max_mu"]

    def forward(self, data: DataDict, q_params) -> OutputDict:
        """Forward pass of the model.

        Args:
            data (DataDict): The input data (or batch). Can be dict or :class:`aml.data.AtomsGraph`.

        Returns:
            OutputDict: The dict of computed outputs.
        """

        if self.canonical:
            results = self.get_q_canonical(
                data, 
                q_params["temperature"] * self.kb,
                q_params["alpha"], 
                q_params["beta"], 
                q_params["dqn"], 
            )
        else:
            results = self.get_q(
                data,
                q_params["temperature"] * self.kb,
                q_params["elem_chempot"],
                q_params["alpha"],
                q_params["beta"],
                q_params["dqn"],
            )
        return results

    def get_q_canonical(
        self,
        data: DataDict,
        kT: Tensor | float,
        alpha=0.0,
        beta=1.0,
        dqn=False,
    ) -> OutputDict:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            kT (float): kT (eV)
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.reaction_model(data)
        outputs[K.reaction_features] = data[K.reaction_features]
        reaction_feat = outputs[K.reaction_features]
        # Calculate q0
        q_0 = -1 * alpha * outputs[K.barrier].unsqueeze(-1) - beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if alpha != 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, dtype=torch.float, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            else:
                raise TypeError("kT type is wrong or not provided")
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        else:
            kT = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        if dqn:
            dqn_feat = F.normalize(torch.cat([q_0, kT * q_1], dim=-1), dim=-1)
            emb = self.Q_emb(reaction_feat)
            total_feat = torch.cat([dqn_feat, emb], dim=-1)
            outputs[K.dqn_feat] = total_feat
            rl_q = self.Q_output(total_feat)
        else:
            rl_q = q_0 + q_1 * kT
        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)

        return outputs

    def get_q(
        self,
        data: DataDict,
        kT: Tensor | float,
        elem_chempot: Dict[str, Tensor | float],
        alpha=0.0,
        beta=1.0,
        dqn=False,
    ) -> OutputDict:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            kT (float): kT (eV)
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.reaction_model(data)
        outputs[K.reaction_features] = data[K.reaction_features]
        reaction_feat = outputs[K.reaction_features]
        # Calculate q0
        q_0 = -1 * alpha * outputs[K.barrier].unsqueeze(-1) - beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if alpha != 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, dtype=torch.float, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            else:
                raise TypeError("kT type is wrong or not provided")
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        else:
            kT = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        # Calculate q2
        elem_chempot_tensor = torch.zeros(len(self.reaction_model.species) - 1, device=outputs[K.delta_e].device)
        q_2_i = beta * self.get_change_n_elems(data)
        for i, specie in enumerate(self.reaction_model.species[1:]):
            if specie in elem_chempot.keys():
                elem_chempot_tensor[i] = torch.tensor(
                    (elem_chempot[specie] - self.mean_mu[specie]) / self.max_mu[specie],
                    device="cuda",
                    dtype=torch.float,
                )
        elem_chempot_tensor = elem_chempot_tensor.repeat(outputs[K.delta_e].shape[0], 1)

        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements
        if dqn:
            dqn_feat = F.normalize(torch.cat([q_0, kT * q_1, elem_chempot_tensor * q_2_i], dim=-1), dim=-1)
            emb = self.Q_emb(reaction_feat)
            total_feat = torch.cat([dqn_feat, emb], dim=-1)
            outputs[K.dqn_feat] = total_feat
            rl_q = self.Q_output(total_feat)
        else:
            q_2 = torch.sum(q_2_i * elem_chempot_tensor, dim=-1, keepdim=True)
            rl_q = q_0 + q_1 * kT + q_2

        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)

        return outputs

@registry.register_model("dqn_v3")
class ReactionDQN3(BaseDQN):
    """DQN model based on the reaction model.

    Args:
        reaction_model (BaseReactionModel): The reaction model.
        N_emb (int, optional): The embedding dimension. Defaults to 16.
        N_feat (int, optional): The hidden dimension. Defaults to 32.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.15.
        canonical (bool, optional): Whether to use the canonical model. Defaults to False.
    """

    def __init__(
        self,
        reaction_model: BaseReactionModel,
        N_emb: int = 16,
        N_feat: int = 32,
        dropout_rate: float = 0.15,
        canonical: bool = False,
        mean_mu: Dict[str, float] | None = None,
        max_mu: Dict[str, float] | None = None,
        threshold: float | None = None,
        gamma: float | None = None,
    ):
        super().__init__(reaction_model)
        self.N_feat = N_feat
        self.N_emb = N_emb
        self.dropout_rate = dropout_rate
        self.canonical = canonical
        self.mean_mu = mean_mu
        self.max_mu = max_mu
        self.threshold = threshold
        self.gamma = gamma
        n_input = N_emb + 2 if canonical else N_emb + len(reaction_model.atomic_numbers) + 1
        self.Q_emb = MLP(
            n_input=reaction_model.reaction_feat + 1,
            n_output=N_emb,
            hidden_layers=(N_emb,),
            activation="leaky_relu",
            w_init="xavier_uniform",
            b_init="zeros",
            dropout_rate=self.dropout_rate,
        )
        self.Q_output = MLP(
            n_input=n_input,
            n_output=1,
            hidden_layers=(N_feat, N_feat),
            activation="leaky_relu",
            w_init="xavier_uniform",
            b_init="zeros",
            dropout_rate=self.dropout_rate,
        )
        self.Q_emb.reset_parameters()
        self.Q_output.reset_parameters()

    def set_chempot_stats(self, chempot_stats):
        self.mean_mu = chempot_stats["mean_mu"]
        self.max_mu = chempot_stats["max_mu"]

    def set_q_threshold(self, threshold: float, gamma: float):
        self.threshold = threshold
        self.gamma = gamma

    def forward(self, data: DataDict, q_params) -> OutputDict:
        """Forward pass of the model.

        Args:
            data (DataDict): The input data (or batch). Can be dict or :class:`aml.data.AtomsGraph`.

        Returns:
            OutputDict: The dict of computed outputs.
        """

        if self.canonical:
            results = self.get_q_canonical(
                data,
                q_params["temperature"] * self.kb,
                q_params["alpha"],
                q_params["beta"],
                q_params["dqn"],
            )
        else:
            results = self.get_q(
                data,
                q_params["temperature"] * self.kb,
                q_params["elem_chempot"],
                q_params["alpha"],
                q_params["beta"],
                q_params["dqn"],
            )
        return results

    def get_q_canonical(
        self,
        data: DataDict,
        kT: Tensor | float,
        alpha=0.0,
        beta=1.0,
        dqn=False,
    ) -> OutputDict:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            kT (float): kT (eV)
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.reaction_model(data)
        outputs[K.reaction_features] = data[K.reaction_features]
        reaction_feat = outputs[K.reaction_features]
        # Calculate q0
        q_0 = -1 * alpha * outputs[K.barrier].unsqueeze(-1) - beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if alpha != 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, dtype=torch.float, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            else:
                raise TypeError("kT type is wrong or not provided")
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        else:
            kT = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        if dqn:
            dqn_feat = F.normalize(torch.cat([q_0, kT * q_1], dim=-1), dim=-1)
            emb = self.Q_emb(reaction_feat)
            total_feat = torch.cat([dqn_feat, emb], dim=-1)
            outputs[K.dqn_feat] = total_feat
            if self.threshold is not None and self.gamma is not None:
                limits = self.threshold / (1-self.gamma)
                rl_q = limits * torch.tanh(self.Q_output(total_feat))
            else:
                rl_q = self.Q_output(total_feat)
        else:
            rl_q = q_0 + q_1 * kT
        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)

        return outputs

    def get_q(
        self,
        data: DataDict,
        kT: Tensor | float,
        elem_chempot: Dict[str, Tensor | float],
        alpha=0.0,
        beta=1.0,
        dqn=False,
    ) -> OutputDict:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            kT (float): kT (eV)
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.reaction_model(data)
        outputs[K.reaction_features] = data[K.reaction_features]
        reaction_feat = outputs[K.reaction_features]
        # Calculate q0
        q_0 = -1 * alpha * outputs[K.barrier].unsqueeze(-1) - beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if alpha != 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, dtype=torch.float, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            else:
                raise TypeError("kT type is wrong or not provided")
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        else:
            kT = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        # Calculate q2
        elem_chempot_tensor = torch.zeros(len(self.reaction_model.species) - 1, device=outputs[K.delta_e].device)
        q_2_i = beta * self.get_change_n_elems(data)
        for i, specie in enumerate(self.reaction_model.species[1:]):
            if specie in elem_chempot.keys():
                elem_chempot_tensor[i] = torch.tensor(
                    (elem_chempot[specie] - self.mean_mu[specie]) / self.max_mu[specie],
                    device="cuda",
                    dtype=torch.float,
                )
        elem_chempot_tensor = elem_chempot_tensor.repeat(outputs[K.delta_e].shape[0], 1)

        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements
        if dqn:
            dqn_feat = F.normalize(torch.cat([q_0, kT * q_1, elem_chempot_tensor * q_2_i], dim=-1), dim=-1)
            emb = self.Q_emb(reaction_feat)
            total_feat = torch.cat([dqn_feat, emb], dim=-1)
            outputs[K.dqn_feat] = total_feat
            if self.threshold is not None and self.gamma is not None:
                limits = self.threshold / (1-self.gamma)
                rl_q = limits * torch.tanh(self.Q_output(total_feat))
            else:
                rl_q = self.Q_output(total_feat)
        else:
            q_2 = torch.sum(q_2_i * elem_chempot_tensor, dim=-1, keepdim=True)
            rl_q = q_0 + q_1 * kT + q_2

        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)

        return outputs
