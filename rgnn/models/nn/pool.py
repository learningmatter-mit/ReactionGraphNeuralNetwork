import torch
from torch import nn
from typing import Optional, Dict, Any
from torch.nn import Sequential
from torch.nn import functional as F
from rgnn.common.registry import registry
from rgnn.models.nn.mlp import MLP


class AttentionPool(nn.Module):
    """
    Compute output quantities using attention, rather than a sum over
    atomic quantities. There are two methods to do this:
    (1): "atomwise": Learn the attention weights from atomic fingerprints,
    get atomwise quantities from a network applied to the fingeprints,
    and sum them with attention weights.
    (2) "graph_fp": Learn the attention weights from atomic fingerprints,
    multiply the fingerprints by these weights, add the fingerprints
    together to get a molecular fingerprint, and put the molecular
    fingerprint through a network that predicts the output.

    This one uses `graph_fp`, since it seems more expressive (?)
    """

    def __init__(
        self,
        feat_dim: int,
        att_act: str,
        prob_func: str = "softmax",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        # graph_fp_act: str,
        # num_out_layers: int,
        # out_dim: int,
        **kwargs,
    ):
        """ """
        super().__init__()

        self.w_mat = nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)

        self.att_weight = torch.nn.Parameter(torch.rand(1, feat_dim))
        activation_kwargs = activation_kwargs or {}
        self.att_act = registry.get_activation_class(att_act)(**activation_kwargs)
        nn.init.xavier_uniform_(self.att_weight, gain=1.414)
        self.prob_func = att_readout_probs(prob_func)

        # # reduce the number of features by the same factor in each layer
        # feat_num = tuple(int(feat_dim / num_out_layers**m) for m in range(num_out_layers))
        # # put together in readout network
        # self.graph_fp_nn = MLP(
        #     n_input=feat_num[0],
        #     n_output=out_dim,
        #     hidden_layers=feat_num[1:],
        #     activation=graph_fp_act,
        #     w_init="xavier_uniform",
        #     b_init="zeros",
        # )

    def forward(self, batch, atomwise_output):
        """
        Args:
            feats (torch.Tensor): n_atom x feat_dim atomic features,
                after convolutions are finished.
        """

        N = batch["n_atoms_i"].detach().cpu().tolist()
        # results = {}
        split_feats = torch.split(atomwise_output, N)
        # all_outputs = []
        learned_feats = []

        for feats in split_feats:
            weights = self.prob_func(self.att_act((self.att_weight * self.w_mat(feats)).sum(-1)))

            graph_fp = (weights.reshape(-1, 1) * self.w_mat(feats)).sum(0)

            # output = self.graph_fp_nn(graph_fp)
            learned_feats.append(graph_fp)
            # learned_feats.append(graph_fp)
        results = torch.stack(learned_feats)
        # results[out_key] = torch.stack(all_outputs).reshape(-1)
        # results[f"{out_key}_features"] = torch.stack(learned_feats)

        return results


def att_readout_probs(name):
    if name.lower() == "softmax":

        def func(output):
            weights = F.softmax(output, dim=0)
            return weights

    elif name.lower() == "square":

        def func(output):
            weights = output**2 / (output**2).sum()
            return weights

    else:
        raise NotImplementedError

    return func
