from typing import TypeVar

import torch
from ase import Atoms
from ase.constraints import FixAtoms

from rgnn.common import keys as K
from rgnn.common.typing import DataDict


T = TypeVar("T")


def maybe_list(x: T | list[T]) -> list[T]:
    """Listify x if it is not a list.
    TODO: Rewrite to use `typing.Sequence` instead of `list`.

    Args:
        x (T | list[T]): The input.

    Returns:
        list[T]: The listified input.
    """
    if isinstance(x, list):
        return x
    else:
        return [x]


def is_pkl(b: bytes) -> bool:
    """Check if the bytes is a pickle file by checking the first two bytes.

    Args:
        b (bytes): The bytes.

    Returns:
        bool: Whether the bytes is a pickle file.
    """
    first_two_bytes = b[:2]
    if first_two_bytes in (b"cc", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return True
    return False


def find_fixatoms_constraint(atoms: Atoms) -> FixAtoms | None:
    """If atoms as FixAtoms contraint, return it.
    Otherwise returns None.

    Args:
        atoms(Atoms): A Atoms object.

    Returns:
        FixAtoms | None
    """
    if not atoms.constraints:
        return None
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            return c
    return None


def compute_neighbor_vecs(data: DataDict) -> DataDict:
    """Compute the vectors between atoms and their neighbors (i->j)
    and store them in ``data[K.edge_vec]``.
    The ``data`` must contain ``data[K.pos]``, ``data[K.edge_index]``, ``data[K.edge_shift]``,
    This function should be called inside ``forward`` since the dependency of neighbor positions
    on atomic positions needs to be tracked by autograd in order to appropriately compute forces.

    Args:
        data (DataDict): The data dictionary.

    Returns:
        DataDict: The data dictionary with ``data[K.edge_vec]``.
    """
    batch = data[K.batch]
    pos = data[K.pos]
    edge_index = data[K.edge_index]  # neighbors
    edge_shift = data[K.edge_shift]  # shift vectors
    batch_size = int((batch.max() + 1).item())
    cell = data[K.cell] if "cell" in data else torch.zeros((batch_size, 3, 3)).to(pos.device)
    idx_i = edge_index[1]
    idx_j = edge_index[0]

    edge_batch = batch[idx_i]  # batch index for edges(neighbors)
    edge_vec = pos[idx_j] - pos[idx_i] + torch.einsum("ni,nij->nj", edge_shift, cell[edge_batch])
    data[K.edge_vec] = edge_vec
    return data


def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, "to") else val
    return gpu_batch


