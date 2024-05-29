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
    """Interatomic potential model.
    It wraps an energy model and computes the energy, force, stress and hessian.
    The energy model should be a subclass of :class:`BaseEnergyModel`.

    Args:
        energy_model (BaseEnergyModel): The energy model which computes the energy from the data.
        compute_force (bool, optional): Whether to compute force. Defaults to True.
        compute_stress (bool, optional): Whether to compute stress. Defaults to eFalse.
        compute_hessian (bool, optional): Whether to compute hessian. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.
    """

    def __init__(
        self,
        reaction_model: BaseReactionModel,
        N_emb: int = 16,
        N_feat: int = 32,
        dropout_rate: float = 0.15,
        canonical: bool = False,
    ):
        super().__init__(reaction_model)
        self.N_feat = N_feat
        self.N_emb = N_emb
        self.dropout_rate = dropout_rate
        self.canonical = canonical

        if canonical:
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
                n_input=N_emb + 2,
                n_output=1,
                hidden_layers=(N_feat, N_feat),
                activation="leaky_relu",
                w_init="xavier_uniform",
                b_init="zeros",
                dropout_rate=self.dropout_rate,
            )
        else:
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
                n_input=N_emb + len(self.reaction_model.atomic_numbers) + 1,
                n_output=1,
                hidden_layers=(N_feat, N_feat),
                activation="leaky_relu",
                w_init="xavier_uniform",
                b_init="zeros",
                dropout_rate=self.dropout_rate,
            )

        self.Q_emb.reset_parameters()
        self.Q_output.reset_parameters()

    def forward(self, data: DataDict, q_params) -> OutputDict:
        """Forward pass of the model.

        Args:
            data (DataDict): The input data (or batch). Can be dict or :class:`aml.data.AtomsGraph`.

        Returns:
            OutputDict: The dict of computed outputs.
        """

        if self.canonical:
            results = self.get_q_canonical(
                data, q_params["alpha"], q_params["beta"], q_params["dqn"], q_params["temperature"] * self.kb
            )
        else:
            results = self.get_q(
                data,
                q_params["alpha"],
                q_params["beta"],
                q_params["dqn"],
                q_params["temperature"] * self.kb,
                q_params["elem_chempot"],
                q_params.get("max_mu", None),
                q_params.get("mean_mu", None),
            )
        return results

    def get_q_canonical(
        self,
        data: DataDict,
        alpha=0.0,
        beta=1.0,
        dqn=False,
        kT: Tensor | float | None = None,
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
        if kT is not None:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            else:
                raise TypeError("kT type is wrong")
            if alpha == 0.0:
                q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            else:
                q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        elif kT is None and alpha != 0.0:
            raise ValueError("kT should be provided")
        else:
            kT = torch.as_tensor(0.0, device=outputs[K.delta_e].device)
            kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
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
        alpha=0.0,
        beta=1.0,
        dqn=False,
        kT: Tensor | float | None = None,
        elem_chempot: Dict[str, Tensor | float] | None = None,
        max_mu: float | None = None,
        mean_mu: float | None = None,
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
        if kT is not None:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, dtype=torch.float, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            else:
                raise TypeError("kT type is wrong")
            if alpha == 0.0:
                q_1 = torch.zeros(
                    outputs[K.delta_e].shape[0], dtype=torch.float, device=outputs[K.delta_e].device
                ).unsqueeze(-1)
            else:
                q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        elif kT is None and alpha != 0.0:
            raise ValueError("kT should be provided")
        else:
            kT = torch.as_tensor(0.0, dtype=torch.float, device=outputs[K.delta_e].device)
            kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        # Calculate q2
        elem_chempot_tensor = torch.zeros(len(self.reaction_model.species) - 1, device=outputs[K.delta_e].device)
        q_2_i = beta * self.get_change_n_elems(data)
        if elem_chempot is not None:
            # TODO: THIS IS VERY HACKY
            check_chempot = elem_chempot[self.reaction_model.species[0]]
            if isinstance(check_chempot, Tensor) and check_chempot.shape[0] > 1:
                batched_elem_chempot_list = []
                for j in range(q_2_i.shape[0]):
                    elem_chempot_tensor_temp = torch.zeros(
                        len(self.reaction_model.species) - 1, device=outputs[K.delta_e].device
                    )
                    for i, specie in enumerate(self.reaction_model.species[1:]):
                        elem_chempot_tensor_temp[i] = torch.tensor(
                            (elem_chempot[specie][j] - mean_mu[specie]) / (max_mu[specie] + 1e-8),
                            device="cuda",
                            dtype=torch.float,
                        )
                    batched_elem_chempot_list.append(elem_chempot_tensor_temp)
                elem_chempot_tensor = torch.stack(batched_elem_chempot_list, dim=0)
            else:
                for i, specie in enumerate(self.reaction_model.species[1:]):
                    elem_chempot_tensor[i] = torch.tensor(
                        (elem_chempot[specie] - mean_mu[specie]) / (max_mu[specie] + 1e-8),
                        device="cuda",
                        dtype=torch.float,
                    )
                elem_chempot_tensor = elem_chempot_tensor.repeat(outputs[K.delta_e].shape[0], 1)
        else:
            elem_chempot_tensor = elem_chempot_tensor.repeat(outputs[K.delta_e].shape[0], 1)
        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements
        if dqn:
            dqn_feat = F.normalize(torch.cat([q_0, kT * q_1, elem_chempot_tensor * q_2_i], dim=-1), dim=-1)
            emb = self.Q_emb(reaction_feat)
            total_feat = torch.cat([dqn_feat, emb], dim=-1)
            outputs[K.dqn_feat] = emb
            rl_q = self.Q_output(total_feat)
        else:
            q_2 = torch.sum(q_2_i * elem_chempot_tensor, dim=-1, keepdim=True)
            rl_q = q_0 + q_1 * kT + q_2

        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)

        return outputs
