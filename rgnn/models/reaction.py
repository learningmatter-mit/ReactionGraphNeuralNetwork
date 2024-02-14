from abc import ABC
from typing import Dict

import torch
from torch.nn import functional as F

from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, OutputDict, Tensor
from rgnn.models.nn.mlp import MLP
from rgnn.models.nn.scale import canocialize_species

from .builder import get_model
from .reaction_models.base import BaseReactionModel


@registry.register_model("reaction_dqn")
class ReactionDQN(torch.nn.Module, ABC):
    """Interatomic potential model.
    It wraps an energy model and computes the energy, force, stress and hessian.
    The energy model should be a subclass of :class:`BaseEnergyModel`.

    Args:
        energy_model (BaseEnergyModel): The energy model which computes the energy from the data.
        compute_force (bool, optional): Whether to compute force. Defaults to True.
        compute_stress (bool, optional): Whether to compute stress. Defaults to False.
        compute_hessian (bool, optional): Whether to compute hessian. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.
    """

    def __init__(
        self,
        reaction_model: BaseReactionModel,
        N_feat: int = 16,
        return_embeddings: bool = True,
    ):
        super().__init__()

        self.reaction_model = reaction_model
        self.N_feat = N_feat
        self.return_embeddings = return_embeddings
        self.cutoff = self.reaction_model.get_cutoff()
        self.Q0_output = MLP(
            n_input=reaction_model.reaction_feat + 2,
            n_output=1,
            hidden_layers=(N_feat, N_feat),
            activation="relu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.Q0_output.reset_parameters()
        self.Q1_output = MLP(
            n_input=reaction_model.reaction_feat + 2,
            n_output=1,
            hidden_layers=(N_feat, N_feat),
            activation="relu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.Q1_output.reset_parameters()
        # TODO: Permutation issue
        self.Q2_i_output = MLP(
            n_input=reaction_model.reaction_feat + len(self.reaction_model.atomic_numbers),
            n_output=len(self.reaction_model.atomic_numbers) - 1,
            hidden_layers=(N_feat, N_feat),
            activation="relu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.Q2_i_output.reset_parameters()

    @torch.jit.unused
    @property
    def output_keys(self) -> tuple[str, ...]:
        keys = [K.delta_e, K.barrier, K.freq, K.rl_q]
        return tuple(keys)

    def forward(self, data: DataDict) -> OutputDict:
        """Forward pass of the model.

        Args:
            data (DataDict): The input data (or batch). Can be dict or :class:`aml.data.AtomsGraph`.

        Returns:
            OutputDict: The dict of computed outputs.
        """

        outputs = self.reaction_model(data)

        if self.return_embeddings:
            for key in self.reaction_model.embedding_keys:
                outputs[key] = data[key]
        return outputs

    def get_q(
        self,
        data: DataDict,
        kT: Tensor | float | None = None,
        elem_chempot: Dict[str, Tensor | float] | None = None,
        alpha=0.0,
        beta=1.0,
        dqn=False,
        # mean=0.0,
        # stddev=1.0,
    ) -> Tensor:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            kT (float): kT (eV)
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.forward(data)
        reaction_feat = data[K.reaction_features]
        # Calculate q0
        q_0 = -1 * alpha * outputs[K.barrier].unsqueeze(-1) - beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if kT is not None and alpha != 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        elif kT is not None and alpha == 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        # TODO: This is not perfect
        else:
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            kT = torch.tensor(0.0, device=outputs[K.delta_e].device)
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
                            elem_chempot[specie][j], device="cuda", dtype=torch.float
                        )
                    batched_elem_chempot_list.append(elem_chempot_tensor_temp)
                elem_chempot_tensor = torch.stack(batched_elem_chempot_list, dim=0)
            else:
                for i, specie in enumerate(self.reaction_model.species[1:]):
                    elem_chempot_tensor[i] = torch.tensor(elem_chempot[specie], device="cuda", dtype=torch.float)
        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements
        if dqn:
            q0_feat = torch.cat([q_0, reaction_feat], dim=-1)
            q1_feat = torch.cat([q_1, reaction_feat], dim=-1)
            q2_feat = torch.cat([q_2_i, reaction_feat], dim=-1)

            q_0 = self.Q0_output(F.normalize(q0_feat, dim=-1))
            q_1 = self.Q1_output(F.normalize(q1_feat, dim=-1))
            q_2_i = self.Q2_i_output(q2_feat)
        q_2 = torch.sum(q_2_i * elem_chempot_tensor, dim=-1, keepdim=True)
        if alpha == 0.0:
            rl_q = q_0 + q_2
        elif beta == 0.0:
            rl_q = q_0 + q_1 * kT
        else:
            rl_q = q_0 + q_1 * kT + q_2
        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)
        outputs[K.Q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.Q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.Q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements, previous version

        return outputs

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
            change_el = torch.zeros(len(self.reaction_model.atomic_numbers), device=data["batch"].device)
            for i, specie in enumerate(unique2):
                elem_change_dict[specie.item()] = counts2[i]
            for i, specie in enumerate(unique1):
                count = elem_change_dict.get(specie.item(), torch.tensor(0, device=data["batch"].device))
                new_count = count - counts1[i]
                elem_change_dict.update({specie.item(): new_count})
            for i, el in enumerate(self.reaction_model.atomic_numbers):
                change_el[i] = elem_change_dict[el]

            change_elem_list[graph_id] = change_el[1:]
        change_elems = torch.stack(change_elem_list, dim=0)
        return change_elems

    def get_change_chempot(self, data, elem_chempot):
        mask_tensor_r = self.reaction_model.create_subgraph_mask(data)
        converted_elem_chempot = {}
        avail_species = []
        for key, val in elem_chempot.items():
            converted_elem_chempot.update(
                {
                    canocialize_species([key])
                    .detach()
                    .item(): torch.tensor(val, device=data["batch"].device, dtype=torch.float)
                }
            )
            avail_species.append(key)

        # chempot_change = torch.zeros(torch.unique(data["batch"]).shape[0], device=data["batch"].device)
        change_q_2_list = [{} for _ in range(torch.unique(data["batch"]).shape[0])]
        for _, graph_id in enumerate(torch.unique(data["batch"])):
            # print(j, graph_id)
            mask = data["batch"] == graph_id
            reactant_elems = data["elems"][mask][mask_tensor_r[mask]]
            product_elems = data["elems"][mask][~mask_tensor_r[mask]]
            # print(reactant_elems[0], product_elems)
            elem_change_dict = {}
            unique2, counts2 = torch.unique(product_elems, return_counts=True)
            unique1, counts1 = torch.unique(reactant_elems, return_counts=True)
            for i, specie in enumerate(unique2):
                elem_change_dict[specie.item()] = counts2[i]
            for i, specie in enumerate(unique1):
                count = elem_change_dict.get(specie.item(), torch.tensor(0, device=data["batch"].device))
                new_count = count - counts1[i]
                elem_change_dict.update({specie.item(): new_count})
            change_q_2_list[graph_id] = elem_change_dict

            change_chempot = 0.0
            for key, change in elem_change_dict.items():
                chempot = converted_elem_chempot[key] * change
                change_chempot += chempot
            change_q_2_list[graph_id] = change_chempot

        return change_q_2_list

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.reaction_model.hyperparams
        N_feat = self.N_feat
        state = {"state_dict": state_dict, "hyper_parameters": hyperparams, "n_feat": N_feat}

        torch.save(state, filename)

    @classmethod
    def load(cls, path: str, return_embeddings: bool = False) -> "ReactionDQN":
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
        # model_config = hparams["hyperparams"]
        state_dict = ckpt["state_dict"]
        N_feat = ckpt.get("n_feat", 16)  # TODO: Should be changed
        # state_dict = {
        #     ".".join(k.split(".")[1:]): v for k, v in state_dict.items()
        # }
        reaction_model = get_model(hparams)
        model = cls(reaction_model, N_feat, return_embeddings)
        model.load_state_dict(state_dict=state_dict)
        return model

    @torch.jit.unused
    @property
    def num_params(self):
        """Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())


@registry.register_model("reaction_dqn2")
class ReactionDQN2(torch.nn.Module, ABC):
    """Interatomic potential model.
    It wraps an energy model and computes the energy, force, stress and hessian.
    The energy model should be a subclass of :class:`BaseEnergyModel`.

    Args:
        energy_model (BaseEnergyModel): The energy model which computes the energy from the data.
        compute_force (bool, optional): Whether to compute force. Defaults to True.
        compute_stress (bool, optional): Whether to compute stress. Defaults to False.
        compute_hessian (bool, optional): Whether to compute hessian. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.
    """

    def __init__(
        self,
        reaction_model: BaseReactionModel,
        N_feat: int = 32,
        N_emb: int = 16,
        return_embeddings: bool = True,
    ):
        super().__init__()

        self.reaction_model = reaction_model
        self.N_feat = N_feat
        self.N_emb = N_emb
        self.return_embeddings = return_embeddings
        self.cutoff = self.reaction_model.get_cutoff()
        self.Q_emb = MLP(
            n_input=reaction_model.reaction_feat + 2 + len(self.reaction_model.atomic_numbers),
            n_output=N_emb,
            hidden_layers=(N_emb,),
            activation="relu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.Q_output = MLP(
            n_input=N_emb + len(self.reaction_model.atomic_numbers),
            n_output=1 + len(self.reaction_model.atomic_numbers),
            hidden_layers=(N_feat, N_feat),
            activation="relu",
            w_init="xavier_uniform",
            b_init="zeros",
        )

        self.Q_emb.reset_parameters()
        self.Q_output.reset_parameters()

    @torch.jit.unused
    @property
    def output_keys(self) -> tuple[str, ...]:
        keys = [K.delta_e, K.barrier, K.freq, K.rl_q]
        return tuple(keys)

    def forward(self, data: DataDict) -> OutputDict:
        """Forward pass of the model.

        Args:
            data (DataDict): The input data (or batch). Can be dict or :class:`aml.data.AtomsGraph`.

        Returns:
            OutputDict: The dict of computed outputs.
        """

        outputs = self.reaction_model(data)

        if self.return_embeddings:
            for key in self.reaction_model.embedding_keys:
                outputs[key] = data[key]
        return outputs

    def get_q(
        self,
        data: DataDict,
        kT: Tensor | float | None = None,
        elem_chempot: Dict[str, Tensor | float] | None = None,
        alpha=0.0,
        beta=1.0,
        dqn=False,
        chempot_threshold=10
        # mean=0.0,
        # stddev=1.0,
    ) -> Tensor:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            kT (float): kT (eV)
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.forward(data)
        reaction_feat = data[K.reaction_features]
        # Calculate q0
        q_0 = -1 * alpha * outputs[K.barrier].unsqueeze(-1) - beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if kT is not None and alpha != 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        elif kT is not None and alpha == 0.0:
            if isinstance(kT, Tensor):
                kT = kT.unsqueeze(-1)
            elif isinstance(kT, float):
                kT = torch.as_tensor(kT, device=outputs[K.delta_e].device)
                kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
        # TODO: This is not perfect
        else:
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device).unsqueeze(-1)
            kT = torch.as_tensor(0.0, device=outputs[K.delta_e].device)
            kT = kT.repeat(outputs[K.delta_e].shape[0], 1)
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
                            elem_chempot[specie][j], device="cuda", dtype=torch.float
                        )
                    batched_elem_chempot_list.append(elem_chempot_tensor_temp)
                elem_chempot_tensor = torch.stack(batched_elem_chempot_list, dim=0)
            else:
                for i, specie in enumerate(self.reaction_model.species[1:]):
                    elem_chempot_tensor[i] = torch.tensor(elem_chempot[specie], device="cuda", dtype=torch.float)
                elem_chempot_tensor = elem_chempot_tensor.repeat(outputs[K.delta_e].shape[0], 1)
        else:
            elem_chempot_tensor = elem_chempot_tensor.repeat(outputs[K.delta_e].shape[0], 1)
        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements
        if dqn:
            q_feat = F.normalize(torch.cat([q_0, q_1], dim=-1), dim=-1)
            q_feat = torch.cat([q_feat, q_2_i, reaction_feat], dim=-1)
            emb = self.Q_emb(q_feat)
            # print(kT.shape, elem_chempot_tensor.shape, emb.shape)
            dqn_feat = torch.cat([kT, elem_chempot_tensor / chempot_threshold, emb], dim=-1)
            outputs[K.dqn_feat] = dqn_feat
            q_out = self.Q_output(dqn_feat)
            q_0 = q_out[:, 0].unsqueeze(-1)
            q_1 = q_out[:, 1].unsqueeze(-1)
            q_2_i = q_out[:, 2:]
        q_2 = torch.sum(q_2_i * elem_chempot_tensor, dim=-1, keepdim=True)
        if alpha == 0.0:
            rl_q = q_0 + q_2
        elif beta == 0.0:
            rl_q = q_0 + q_1 * kT
        else:
            rl_q = q_0 + q_1 * kT + q_2
        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)
        outputs[K.Q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.Q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.Q2_i] = q_2_i  # (N, M) --> N: batch, M: number of elements, previous version

        return outputs

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
            change_el = torch.zeros(len(self.reaction_model.atomic_numbers), device=data["batch"].device)
            for i, specie in enumerate(unique2):
                elem_change_dict[specie.item()] = counts2[i]
            for i, specie in enumerate(unique1):
                count = elem_change_dict.get(specie.item(), torch.tensor(0, device=data["batch"].device))
                new_count = count - counts1[i]
                elem_change_dict.update({specie.item(): new_count})
            for i, el in enumerate(self.reaction_model.atomic_numbers):
                change_el[i] = elem_change_dict[el]

            change_elem_list[graph_id] = change_el[1:]
        change_elems = torch.stack(change_elem_list, dim=0)
        return change_elems

    def get_change_chempot(self, data, elem_chempot):
        mask_tensor_r = self.reaction_model.create_subgraph_mask(data)
        converted_elem_chempot = {}
        avail_species = []
        for key, val in elem_chempot.items():
            converted_elem_chempot.update(
                {
                    canocialize_species([key])
                    .detach()
                    .item(): torch.tensor(val, device=data["batch"].device, dtype=torch.float)
                }
            )
            avail_species.append(key)

        # chempot_change = torch.zeros(torch.unique(data["batch"]).shape[0], device=data["batch"].device)
        change_q_2_list = [{} for _ in range(torch.unique(data["batch"]).shape[0])]
        for _, graph_id in enumerate(torch.unique(data["batch"])):
            # print(j, graph_id)
            mask = data["batch"] == graph_id
            reactant_elems = data["elems"][mask][mask_tensor_r[mask]]
            product_elems = data["elems"][mask][~mask_tensor_r[mask]]
            # print(reactant_elems[0], product_elems)
            elem_change_dict = {}
            unique2, counts2 = torch.unique(product_elems, return_counts=True)
            unique1, counts1 = torch.unique(reactant_elems, return_counts=True)
            for i, specie in enumerate(unique2):
                elem_change_dict[specie.item()] = counts2[i]
            for i, specie in enumerate(unique1):
                count = elem_change_dict.get(specie.item(), torch.tensor(0, device=data["batch"].device))
                new_count = count - counts1[i]
                elem_change_dict.update({specie.item(): new_count})
            change_q_2_list[graph_id] = elem_change_dict

            change_chempot = 0.0
            for key, change in elem_change_dict.items():
                chempot = converted_elem_chempot[key] * change
                change_chempot += chempot
            change_q_2_list[graph_id] = change_chempot

        return change_q_2_list

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.reaction_model.hyperparams
        state = {"state_dict": state_dict, "hyper_parameters": hyperparams, "n_feat": self.N_feat, "n_emb": self.N_emb}

        torch.save(state, filename)

    @classmethod
    def load(cls, path: str, return_embeddings: bool = False) -> "ReactionDQN":
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
        # model_config = hparams["hyperparams"]
        state_dict = ckpt["state_dict"]
        N_feat = ckpt.get("n_feat", 16)  # TODO: Should be changed
        N_emb = ckpt.get("n_emb", 16)  # TODO: Should be changed
        # state_dict = {
        #     ".".join(k.split(".")[1:]): v for k, v in state_dict.items()
        # }
        reaction_model = get_model(hparams)
        model = cls(reaction_model, N_feat, N_emb, return_embeddings)
        model.load_state_dict(state_dict=state_dict)
        return model

    @torch.jit.unused
    @property
    def num_params(self):
        """Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())
