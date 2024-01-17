from typing import Dict, Sequence

import torch
from torch_geometric import typing as pyg_typing

Tensor = torch.Tensor
OptTensor = pyg_typing.OptTensor

DataDict = Dict[str, Tensor]
OutputDict = Dict[str, Tensor]
Species = Sequence[int] | Sequence[str]
