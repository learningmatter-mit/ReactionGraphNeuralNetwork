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


@registry.register_model("p_net")
class PNet(torch.nn.Module, Configurable, ABC):
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
        n_feat: int = 32,
        dropout_rate: float = 0.15,
        threshold: float | None = None,
        gamma: float | None = None,
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
        self.probs_out = MLP(
            n_input=hidden_channels + 2,
            n_output=1,
            hidden_layers=(self.n_feat, self.n_feat),
            activation="leaky_relu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.kb = 8.617 * 10**-5
        self.threshold = threshold
        self.gamma = gamma
        self.reset_parameters()

    def reset_parameters(self):
        self.representation.reset_parameters()
        self.probs_out.reset_parameters()

    def forward(self, data: DataDict, kT: Tensor | float, actions: Tensor | None = None) -> DataDict:
        compute_neighbor_vecs(data)
        if isinstance(kT, Tensor):
            kT = kT.unsqueeze(-1)
        elif isinstance(kT, float):
            kT = torch.as_tensor(kT, dtype=torch.float, device=data["n_atoms"].device)
            kT = kT.repeat(data["elems"].shape[0], 1)
        else:
            raise TypeError("kT type is wrong or not provided")
        node_degrees = degree(data["edge_index"][0], num_nodes=torch.sum(data["n_atoms"]))
        new_features_list = []
        # new_features_list_2 = []
        for graph_id in torch.unique(data["batch"]):
            mask = data["batch"] == graph_id
            filtered_nodes = node_degrees[mask]

            _, sorted_index = torch.sort(filtered_nodes, dim=-1, stable=True)
            new_features_list.append(sorted_index)
        connectivity_feat = torch.concat(new_features_list, dim=0)
        # data.update({K.elems: connectivity_feat})
        data = self.representation(data)
        mask_tensor_r = self.create_subgraph_mask(data)
        probability = self.probs_out(
            torch.cat([kT, connectivity_feat.unsqueeze(-1), data[K.node_features]], dim=-1)
        )[mask_tensor_r]

        limits = self.threshold / (1-self.gamma)
        rl_q = limits * torch.tanh(probability)
        rl_q = rl_q.squeeze(-1)
        #This can be only done for single data

        if "focal_actions" in data.keys(): # Training (batch data)
            # Verify focal_actions length matches batch size
            assert len(data["focal_actions"]) == len(data["n_atoms_i"]), \
                f"Focal actions length ({len(data['focal_actions'])}) doesn't match batch size ({len(data['n_atoms_i'])})"
            
            # Calculate cumulative offsets
            cumsum_atoms = torch.cat([
                torch.tensor([0], device=rl_q.device),
                torch.cumsum(data["n_atoms_i"][:-1], dim=0)
            ])
            # Verify indices are within bounds
            # print(data["focal_actions"], data["num_atoms_i"])
            assert torch.all(data["focal_actions"] < data["n_atoms_i"]), \
                "Focal action indices exceed number of atoms in graphs"
            
            # Calculate batch-adjusted indices
            batch_focal_actions = data["focal_actions"] + cumsum_atoms
            # print(batch_focal_actions)
            outputs = {"rl_q": rl_q[batch_focal_actions]}
        else: # Inference one data
            softmaxed_probs = F.softmax(rl_q / kT[mask_tensor_r].squeeze(-1), dim=-1)
            outputs = {"focal_p": softmaxed_probs, "rl_q": rl_q}
        # softmaxed_probabilities = torch.stack(softmaxed_probabilities_list, dim=0)

        return outputs

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

    def set_q_threshold(self, threshold: float, gamma: float):
        self.threshold = threshold
        self.gamma = gamma

    def create_subgraph_mask(self, data):
        """
        Create a mask tensor for the initial subgraphs in each data point of the batch.

        :param batch: Tensor indicating the data point each node belongs to.
        :param n_atoms_i: Tensor with the number of nodes in the initial subgraph of each data point.
        :param n_atoms_f: Tensor with the number of nodes in the final subgraph of each data point.
        :return: Mask tensor indicating nodes belonging to the initial subgraphs.
        """
        # Initialize mask tensor of the same size as batch, filled with False
        mask_tensor = torch.zeros_like(data[K.batch], dtype=torch.bool, device=data[K.batch].device)

        # Iterate through each data point in the data[K.batch]
        for data_point in torch.unique(data[K.batch]):
            # Find the indices where this data point appears in the data[K.batch]
            indices = (data[K.batch] == data_point).nonzero(as_tuple=True)[0]

            # Calculate the start and end index for the initial subgraph of this data point
            start_idx = indices[0]
            end_idx = start_idx + data[K.n_atoms_i][data_point]

            # Set the corresponding elements in the mask tensor to True
            mask_tensor[start_idx:end_idx] = 1

        return mask_tensor
