import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import ase.neighborlist
import matscipy.neighbours
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


## Copied from "https://github.com/hans-group/aml"

def distance_to_face(x: np.ndarray, face_vec_1: np.ndarray, face_vec_2: np.ndarray) -> float:
    """Compute the distance between a point and a face of a parallelepiped.

    Args:
        x: The point.
        face_vec_1: The first vector of the face.
        face_vec_2: The second vector of the face.

    Returns:
        The distance between the point and the face.
    """
    face_normal_vec = np.cross(face_vec_1, face_vec_2)
    face_normal_vec /= np.linalg.norm(face_normal_vec)
    return np.abs(np.dot(face_normal_vec, x))


def minimum_distance_to_cell(x: Tensor, cell: Tensor) -> float:
    """Compute the minimum distance between a point and a cell.

    Args:
        x: The point.
        cell: The cell.

    Returns:
        The minimum distance between the point and the cell.
    """
    vec_a = cell[0]
    vec_b = cell[1]
    vec_c = cell[2]
    face_dist_ab = distance_to_face(x, vec_a, vec_b)
    face_dist_bc = distance_to_face(x, vec_b, vec_c)
    face_dist_ca = distance_to_face(x, vec_c, vec_a)
    face_dist_min = min(face_dist_ab, face_dist_bc, face_dist_ca)
    return face_dist_min


class NeighborList(NamedTuple):
    """The namedtuple that stores neighbor list.

    Attributes:
        center_idx: The indices of center atoms.
        neighbor_idx: The indices of neighbor atoms.
        offset: The offset of neighbor atoms.
    """

    center_idx: Tensor
    neighbor_idx: Tensor
    offset: Tensor


class NeighborListBuilder(ABC):
    """Base class for neighbor list builder.

    Args:
        cutoff (float): Cutoff radius for neighbor list.
        self_interaction (bool): Whether to include self interaction. Default: False.

    """

    def __init__(self, cutoff: float, self_interaction: bool = False):
        self.cutoff = cutoff
        self.self_interaction = self_interaction

    @abstractmethod
    def build(self, atoms_graph: Data) -> NeighborList:
        """Build neighbor list for given atoms."""


class ASENeighborListBuilder(NeighborListBuilder):
    """ASE neighbor list builder.
    Usually the slowest one, but only one that supports large cutoff for small unit cell.

    Args:
        cutoff (float): Cutoff radius for neighbor list.
        self_interaction (bool): Whether to include self interaction. Default: False.

    """

    def build(self, atoms_graph: Data) -> NeighborList:
        if atoms_graph.volume() == 0:
            pbc = np.array([False, False, False])
        else:
            pbc = np.array([True, True, True])
        device = atoms_graph.pos.device
        pos = atoms_graph.pos.cpu().detach().numpy().astype(np.float64)
        cell = atoms_graph.cell.squeeze().cpu().detach().numpy().astype(np.float64)
        elems = atoms_graph.elems.cpu().detach().numpy().astype(np.int32)

        center_idx, neighbor_idx, offset = ase.neighborlist.primitive_neighbor_list(
            "ijS", pbc, cell, pos, self.cutoff, elems, self_interaction=self.self_interaction
        )
        return NeighborList(
            center_idx=torch.LongTensor(center_idx).to(device),
            neighbor_idx=torch.LongTensor(neighbor_idx).to(device),
            offset=torch.as_tensor(offset, dtype=torch.float).to(device),
        )


class MatscipyNeighborListBuilder(NeighborListBuilder):
    """Matscipy neighbor list builder.
    Fast for both periodic and non-periodic systems.
    Uses minimum image convention (MIC) for periodic systems, so
    the neighbor list may be wrong if the cutoff is larger than the cell.

    Args:
        cutoff (float): Cutoff radius for neighbor list.
        self_interaction (bool): Whether to include self interaction. Default: False.

    """

    def build(self, atoms_graph: Data) -> NeighborList:
        if atoms_graph.volume() == 0:
            pbc = np.array([False, False, False])
        else:
            pbc = np.array([True, True, True])
        device = atoms_graph.pos.device
        # matscipy.neighbours.neighbour_list fails for non-periodic systems
        pos = atoms_graph.pos.cpu().detach().numpy().astype(np.float64)
        cell = atoms_graph.cell.squeeze().cpu().detach().numpy().astype(np.float64)
        elems = atoms_graph.elems.cpu().detach().numpy().astype(np.int32)
        if not pbc.all():
            # put atoms in a box with periodic boundary conditions
            rmin = np.min(pos, axis=0)
            rmax = np.max(pos, axis=0)
            celldim = np.max(rmax - rmin) + 2.5 * self.cutoff
            cell = np.eye(3) * celldim
        else:
            cell_center = np.sum(cell, axis=0) / 2
            min_cell_dist = minimum_distance_to_cell(cell_center, cell)
            if min_cell_dist < self.cutoff:
                warnings.warn(
                    "Cutoff is larger than the minimum distance to the cell. "
                    "It may break MIC and return wrong neighbor lists.",
                    stacklevel=1,
                )
        center_idx, neighbor_idx, offset = matscipy.neighbours.neighbour_list(
            "ijS", cutoff=self.cutoff, positions=pos, pbc=pbc, cell=cell, numbers=elems
        )
        # add self interaction as ase.neighborlist does
        if self.self_interaction:
            center_idx = np.concatenate([center_idx, np.arange(len(pos))])
            neighbor_idx = np.concatenate([neighbor_idx, np.arange(len(pos))])
            offset = np.concatenate([offset, np.zeros((len(pos), 3))])
            # sort by center_idx
            idx = np.argsort(center_idx)
            center_idx = center_idx[idx]
            neighbor_idx = neighbor_idx[idx]
            offset = offset[idx]

        return NeighborList(
            center_idx=torch.LongTensor(center_idx).to(device),
            neighbor_idx=torch.LongTensor(neighbor_idx).to(device),
            offset=torch.as_tensor(offset, dtype=torch.float).to(device),
        )



_neighborlistbuilder_cls_map = {
    "ase": ASENeighborListBuilder,
    "matscipy": MatscipyNeighborListBuilder
}


def resolve_neighborlist_builder(neighborlist_backend: Union[str, object]) -> type:
    """Resolve neighbor list builder by backend name.

    Args:
        neighborlist_backend (str or object): The backend name or the backend class.

    Returns:
        The backend class.
    """
    if isinstance(neighborlist_backend, str):
        return _neighborlistbuilder_cls_map[neighborlist_backend]
    return neighborlist_backend
