from abc import ABC
from typing import Dict

import torch
from torch.nn import functional as F

from rgnn.common import keys as K
from rgnn.common.typing import DataDict, OutputDict, Tensor
from rgnn.models.nn.mlp import MLP
from rgnn.models.nn.scale import ScaleShift, canocialize_species

from .builder import get_model
from .reaction_models.base import BaseReactionModel
from .registry import registry


@registry.register_model("reaction_NN")
class ReactionGNN(torch.nn.Module, ABC):
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
        return_embeddings: bool = True,
    ):
        super().__init__()

        self.reaction_model = reaction_model
        self.return_embeddings = return_embeddings
        self.cutoff = self.reaction_model.get_cutoff()
        # self.dqn = dqn
        # if self.dqn:
        # Initialize q_q params
        # TODO: Permutation issue
        self.q_2_output = MLP(
            n_input=reaction_model.reaction_feat,
            n_output=len(self.reaction_model.species),
            hidden_layers=(reaction_model.reaction_feat, reaction_model.reaction_feat),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.q_2_output.reset_parameters()
        self.q_2_scaler = ScaleShift()
        self.rl_q_output = MLP(
            n_input=reaction_model.reaction_feat + 2,
            n_output=3,
            hidden_layers=(reaction_model.reaction_feat, reaction_model.reaction_feat),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.rl_q_output.reset_parameters()
        self.probs_out = MLP(
            n_input=reaction_model.hidden_channels,
            n_output=1,
            hidden_layers=(reaction_model.hidden_channels // 2, reaction_model.hidden_channels // 2),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.probs_out.reset_parameters()

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

        outputs = {}
        energy_r, energy_p, barrier, freq = self.reaction_model(data)
        # outputs[K.energy_i] = energy_r
        # outputs[K.energy_f] = energy_p
        delta_e = energy_p - energy_r
        if self.reaction_model.scale_output:
            delta_e = self.reaction_model.scale_shift(K.delta_e, delta_e)

        outputs[K.delta_e] = delta_e
        outputs[K.barrier] = barrier
        outputs[K.freq] = freq

        if self.return_embeddings:
            for key in self.reaction_model.embedding_keys:
                outputs[key] = data[key]
        return outputs

    def get_p(self, data: DataDict):
        mask_tensor_r = self.create_subgraph_mask(data)
        _ = self.forward(data)
        softmaxed_probabilities = []
        probability = self.probs_out(data[K.node_features])[mask_tensor_r].squeeze(-1)
        for graph_id in torch.unique(data["batch"]):
            mask = data["batch"] == graph_id
            graph_probs = probability[mask[mask_tensor_r]]
            softmaxed_probs = F.softmax(graph_probs, dim=-1)
            softmaxed_probabilities.append(softmaxed_probs)

        return softmaxed_probabilities

    def get_q(
        self,
        data: DataDict,
        temperature: float | None = None,
        elem_chempot: Dict[str, float] | None = None,
        alpha=0.0,
        beta=1.0,
        dqn=False,
    ) -> Tensor:
        """Calculate the Q for a given reaction graph

        Args:
            data (DataDict): Reaction graph
            temperature (float): kT
            alpha (float, optional): kinetic part of the q. Defaults to 0.0.
            beta (float, optional): thermodynamic part of the q. Defaults to 1.0.

        Returns:
            Total Q (Tensor):  Total Q value
        """
        outputs = self.forward(data)
        reaction_feat = data[K.reaction_features]
        # Calculate q0
        q_0 = alpha * outputs[K.barrier].unsqueeze(-1) + beta * outputs[K.delta_e].unsqueeze(-1)
        # Calculate q1
        if temperature is not None:
            q_1 = alpha * outputs[K.freq].unsqueeze(-1)
        else:
            q_1 = torch.zeros(outputs[K.delta_e].shape[0], device=outputs[K.delta_e].device)
        # Calculate q2
        elem_chempot_tensor = torch.zeros(len(self.reaction_model.species), device=outputs[K.delta_e].device)
        q_2 = self.q_2_output(reaction_feat)
        if elem_chempot is not None:
            for i, specie in enumerate(self.reaction_model.species):
                elem_chempot_tensor[i] = torch.tensor(elem_chempot[specie], device="cuda", dtype=torch.float)

        q_2_mu = q_2 * elem_chempot_tensor
        q_2_mu = torch.sum(q_2_mu, dim=-1, keepdim=True)

        if dqn:
            reaction_feat = torch.cat([q_0, q_1, q_2_mu, reaction_feat], dim=-1)
            reaction_feat = F.normalize(reaction_feat, dim=-1)
            q_total = self.rl_q_output(reaction_feat)
            q_0, q_1, q_2_mu = torch.chunk(q_total, 3, dim=-1)
            # print(q_1.shape, q_2.shape)
            # print(elem_chempot.shape)
            # q_2 = q_2*elem_chempot_tensor
            # print(q_2.shape)
            rl_q = q_0 + q_1 * temperature + q_2_mu
        else:
            rl_q = q_0 + q_1 * temperature + q_2_mu
        # print(rl_q.shape, q_0.shape, q_1.shape, q_2.shape)
        outputs[K.rl_q] = rl_q.squeeze(-1)  # (N,)
        outputs[K.q0] = q_0.squeeze(-1)  # (N,)
        outputs[K.q1] = q_1.squeeze(-1)  # (N,)
        outputs[K.q2] = q_2  # (N, M) --> N: batch, M: number of elements

        return outputs

    def get_change_chempot(self, data, mask_tensor_r, elem_chempot):
        converted_elem_chempot = {}
        for key, val in elem_chempot.items():
            converted_elem_chempot.update(
                {canocialize_species([key]).detach().item(): torch.tensor(val, device="cuda", dtype=torch.float)}
            )
        q_prime = torch.zeros(torch.unique(data["batch"]).shape[0], device="cuda")
        for j, graph_id in enumerate(torch.unique(data["batch"])):
            mask = data["batch"] == graph_id
            # print(mask.shape, mask_tensor_r.shape, batch["elems"][mask].shape)
            # print(mask[mask_tensor_r])
            reactant_elems = data["elems"][mask][mask_tensor_r[mask]]
            # print(reactant_elems.shape, batch["n_atoms_i"])
            # reactant_elems = torch.split(reactant_elems, batch["n_atoms_i"])
            product_elems = data["elems"][mask][~mask_tensor_r[mask]]
            # product_elems = torch.split(product_elems, batch["n_atoms_f"])
            for i, reactant_elem in enumerate(reactant_elems):
                unique1, counts1 = torch.unique(reactant_elem, return_counts=True)
                unique2, counts2 = torch.unique(product_elems[i], return_counts=True)

                # Convert to Python dictionaries for easy comparison
                count_dict1 = dict(zip(unique1.cpu().numpy(), counts1.cpu().numpy()))
                count_dict2 = dict(zip(unique2.cpu().numpy(), counts2.cpu().numpy()))
                change_chempot = 0.0
                # Compare the counts
                for item in set(count_dict1.keys()).union(count_dict2.keys()):
                    count1 = count_dict1.get(item, 0)
                    count2 = count_dict2.get(item, 0)
                    change = count2 - count1
                    chempot = converted_elem_chempot[item] * change
                    # print(chempot)
                    change_chempot += chempot
                    # print(f"Item {item}: Count in tensor1 = {count1}, Count in tensor2 = {count2}, Change = {count2 - count1}")
                q_prime[j] = change_chempot
        return q_prime

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = (self.state_dict(),)
        hyperparams = (self.reaction_model.hyperparams,)
        state = {"state_dict": state_dict, "hyper_parameters": hyperparams}

        torch.save(state, filename)

    # "best_metric": best_metric,
    @torch.jit.unused
    @classmethod
    def load(cls, path: str, return_embeddings: bool = False) -> "ReactionGNN":
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
        # state_dict = {
        #     ".".join(k.split(".")[1:]): v for k, v in state_dict.items()
        # }
        reaction_model = get_model(hparams)
        model = cls(reaction_model, return_embeddings)
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
