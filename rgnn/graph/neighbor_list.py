import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import ase.neighborlist
import matscipy.neighbours
import numpy as np
import torch
from torch import Tensor
from torch_cluster.radius import radius, radius_graph
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


class TorchNeighborListBuilder(NeighborListBuilder):
    """Pytorch neighbor list builder.
    Note that this builder also uses minimum image convention,
    so cutoff should be smaller than half of the box size.

    Args:
        cutoff (float): Cutoff radius for neighbor list.
        self_interaction (bool): Whether to include self interaction. Default: False.
        max_num_neighbors (int): Maximum number of neighbors. Default: 64.

    """

    def __init__(self, cutoff: float, self_interaction: bool = False, max_num_neighbors: int = 64):
        self.cutoff = cutoff
        self.self_interaction = self_interaction
        self.max_num_neighbors = max_num_neighbors

    def build(self, atoms_graph: Data) -> NeighborList:
        vol = atoms_graph.cell.squeeze().det().item()
        pbc = vol > 1e-8
        if pbc:
            cell = atoms_graph.cell.squeeze().cpu().numpy()
            cell_center = np.sum(cell, axis=0) / 2
            min_cell_dist = minimum_distance_to_cell(cell_center, cell)
            if min_cell_dist < self.cutoff:
                warnings.warn(
                    "Cutoff is larger than the minimum distance to the cell. "
                    "It may break MIC and return wrong neighbor lists.",
                    stacklevel=1,
                )
            edge_index, edge_shift = radius_graph_pbc(
                atoms_graph.pos, self.cutoff, atoms_graph.cell, max_num_neighbors=self.max_num_neighbors
            )
        else:
            edge_index = radius_graph(
                atoms_graph.pos, self.cutoff, None, None, max_num_neighbors=self.max_num_neighbors
            )
            edge_shift = torch.zeros((edge_index.size(1), 3), dtype=torch.float32, device=edge_index.device)
        idx_neighbor, idx_center = edge_index[0], edge_index[1]
        return NeighborList(idx_center, idx_neighbor, edge_shift)


def radius_graph_pbc(pos: Tensor, cutoff: Tensor, cell: Tensor, max_num_neighbors: int = 64) -> tuple[Tensor, Tensor]:
    """PBC version of torch_cluster.radius_graph.

    Args:
        pos: The positions of atoms.
        cutoff: The cutoff radius.
        cell: The cell.
        max_num_neighbors: Maximum number of neighbors. Default: 64.

    Returns:
        The edge index and the edge shift.
    """
    assert cell.norm(dim=-1).max() < 1e4  # due to the precision problem
    cell = cell.squeeze()

    vol = cell.det()
    area = torch.cross(cell.roll(shifts=1, dims=0), cell.roll(shifts=2, dims=0), dim=1).norm(dim=1)
    height = vol / area

    # to consider atoms out of the cell
    extra_R = (pos @ cell.inverse()).floor_divide(1.0)

    bound = (cutoff / height).ceil()
    l, m, n = -bound + extra_R.min()
    L, M, N = bound + extra_R.max() + 1.0  # plus 1 due to the boundary [,) in torch.arange below

    grid_l = torch.arange(l.item(), L.item(), device=pos.device)
    grid_m = torch.arange(m.item(), M.item(), device=pos.device)
    grid_n = torch.arange(n.item(), N.item(), device=pos.device)
    mesh_lmn = torch.stack(torch.meshgrid(grid_l, grid_m, grid_n, indexing="ij")).view(3, -1).transpose(0, 1)

    R = mesh_lmn @ cell
    R_pos = (R.unsqueeze(1) + pos.unsqueeze(0)).view(
        -1, 3
    )  # (num_R, num_pos, 3) -> (num_pos*num_R, 3) not (num_R*num_pos, 3)

    row, col = radius(pos, R_pos, cutoff, None, None, max_num_neighbors=max_num_neighbors)  # row: R_pos, col: pos
    pos_row, pos_col = R_pos[row], pos[col]
    row, lmn_row = row.remainder(pos.size(0)), row.floor_divide(pos.size(0))

    mask = (row != col) | (pos_row != pos_col).any(dim=1)
    row, col, lmn_row = row[mask], col[mask], lmn_row[mask]

    edge_index = torch.stack([col, row], dim=0)
    edge_shift = -mesh_lmn[lmn_row]

    return edge_index, edge_shift


_neighborlistbuilder_cls_map = {
    "ase": ASENeighborListBuilder,
    "matscipy": MatscipyNeighborListBuilder,
    "torch": TorchNeighborListBuilder,
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
