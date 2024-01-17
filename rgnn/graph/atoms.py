"""
Data structure for constructing dataset for atomistic machine learning.
"""
from typing import Sequence, Union

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import voigt_6_to_full_3x3_stress
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, Tensor

from .neighbor_list import NeighborListBuilder, resolve_neighborlist_builder
from .utils import find_fixatoms_constraint

IndexType = Union[slice, Tensor, np.ndarray, Sequence]
_default_dtype = torch.get_default_dtype()


class AtomsGraph(Data):
    """Basic graph representation of an atomic system.
    The arguments below are all optional, but commonly used.

    Args:
        elems (Tensor): 1D tensor of atomic numbers.
        pos (Tensor): 2D tensor of atomic positions. (N, 3)
        cell (OptTensor, optional): 1D or 2D tensor of lattice vectors. (3, 3) or (1, 3, 3)
            Automatically unsqueeze to 2D if 1D is given.
        edge_index (OptTensor, optional): Edge index. Defaults to None.
            If this means neighbor indices, 0th row is neighbor and 1st row is center.
            This is because message passing occurs from neighbors to centers.
        edge_shift (OptTensor, optional): Optional shift vectors when creating neighbor list.
            This is non-zero when PBC is applied.
            Defaults to None.
        energy (OptTensor, optional): Energy of the system. Defaults to None.
        force (OptTensor, optional): Force on each atom. Defaults to None.
        node_features (OptTensor, optional): Node features. Defaults to None.
        edge_features (OptTensor, optional): Edge features. Defaults to None.
        global_features (OptTensor, optional): Global features. Defaults to None.
        add_batch (bool, optional): If True, add batch index to the graph. Defaults to False.
    """

    def __init__(
        self,
        elems: Tensor = None,
        pos: Tensor = None,
        cell: OptTensor = None,
        edge_index: OptTensor = None,
        edge_shift: OptTensor = None,
        energy: OptTensor = None,
        force: OptTensor = None,
        stress: OptTensor = None,
        node_features: OptTensor = None,
        edge_features: OptTensor = None,
        global_features: OptTensor = None,
        node_vec_features: OptTensor = None,
        edge_vec_features: OptTensor = None,
        global_vec_features: OptTensor = None,
        fixed_atoms: OptTensor = None,
        add_batch: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.elems = elems
        self.pos = pos
        self.cell = cell
        self.edge_index = edge_index
        self.edge_shift = edge_shift
        self.energy = energy
        self.force = force
        self.stress = stress
        self.node_features = node_features
        self.edge_features = edge_features
        self.global_features = global_features
        self.node_vec_features = node_vec_features
        self.edge_vec_features = edge_vec_features
        self.global_vec_features = global_vec_features
        self.fixed_atoms = fixed_atoms
        if elems is not None:
            self.n_atoms = torch.tensor([elems.size(0)], dtype=torch.long)

        if pos is not None and elems is not None:
            assert pos.shape[0] == elems.shape[0], "Number of atoms and number of positions must be same."
        if cell is None and pos is not None:
            self.cell = torch.zeros((1, 3, 3), dtype=_default_dtype)
        if cell is not None:
            if cell.ndim == 2:
                self.cell = cell.unsqueeze(0)
        if energy is not None and energy.ndim == 0:
            self.energy = energy.unsqueeze(0)

        if add_batch:
            self.batch = torch.zeros_like(elems, dtype=torch.long, device=pos.device)

    @classmethod
    def from_ase(
        cls,
        atoms: Atoms,
        neighborlist_cutoff: float | None = None,
        self_interaction: bool = False,
        energy: float = None,
        force: Tensor = None,
        stress: Tensor = None,
        *,
        read_properties: bool = True,
        add_batch: bool = True,
        neighborlist_backend: Union[str, NeighborListBuilder] = "ase",
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Create AtomsGraph from ASE Atoms object.
        Optionally, neighborlist can be built automatically.

        Args:
            atoms (Atoms): An ASE Atoms object.
            build_neighbors (bool, optional): Whether to build neighborlist or not. Defaults to False.
            cutoff (float, optional): Cutoff radius for neighbor list in Angstrom. Defaults to 5.0.
            self_interaction (bool, optional): Whether to add atom as neighbor of itself(=self loop). Defaults to False.
            energy (float, optional): Potential energy of the system. When set to None, energy will retrieved from
                    atoms.calc if available. Defaults to None.
            force (Tensor, optional): Interatomic forces of the system. When set to None, forces will retrieved from
                    atoms.calc if available. Defaults to None.
            add_batch (bool, optional): Whether to add batch index as attribute or not. Defaults to True.
            neighborlist_backend (Union[str, NeighborListBuilder], optional): The backend for building neighborlist.
                    Accepts `str` or `NeighborListBuilder` class. See `aml.data.neighbor_list.py`.
                    Usually "torch" is the fastest, but the speed is simliary to "matscipy" when PBC is used.
                    "matscipy" is also fast, but it uses minimal image convention,
                        so cannot be used for small cell.
                    "ase" is the slowest, but it can be used on any system.
                    Defaults to "ase".
        Returns:
            AtomsGraph: AtomsGraph object.
        """
        device = "cpu" if device is None else device
        elems = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
        pos = torch.tensor(atoms.positions, dtype=_default_dtype, device=device)
        cell = cls.resolve_cell(atoms).to(device)
        n_atoms = torch.tensor([len(atoms)], dtype=torch.long, device=device)
        if read_properties:
            if energy is None:
                try:
                    energy = atoms.get_potential_energy()
                    energy = torch.as_tensor(energy, dtype=_default_dtype, device=device)
                except RuntimeError:
                    pass
            if force is None:
                try:
                    force = atoms.get_forces()
                    force = torch.as_tensor(force, dtype=_default_dtype, device=device)
                except RuntimeError:
                    pass
            if stress is None:
                try:
                    stress = atoms.get_stress(voigt=False)
                    stress = torch.as_tensor(np.array([stress]), dtype=_default_dtype, device=device)
                except RuntimeError:
                    pass
            else:
                if stress.shape == (6,):
                    stress = voigt_6_to_full_3x3_stress(stress)
                stress = torch.as_tensor(stress, dtype=_default_dtype, device=device)

        atoms_graph = cls(
            elems, pos, cell, None, None, energy, force, stress, n_atoms=n_atoms, add_batch=add_batch, **kwargs
        )

        fixatom = find_fixatoms_constraint(atoms)
        if fixatom is not None:
            fixatoms_constraints = torch.as_tensor(fixatom.index, dtype=_default_dtype, device=device)
            atoms_graph.fixed_atoms = fixatoms_constraints
            atoms_graph.n_fixed_atoms = torch.tensor([fixatoms_constraints.size(0)], dtype=torch.long)

        if neighborlist_cutoff is not None:
            atoms_graph.build_neighborlist(neighborlist_cutoff, self_interaction, neighborlist_backend)
        return atoms_graph

    def build_neighborlist(
        self,
        cutoff: float,
        self_interaction: bool = False,
        backend: Union[str, NeighborListBuilder] = "ase",
    ):
        """Build neighborlist.

        Args:
            cutoff (float): Cutoff radius for neighbor list in Angstrom.
            self_interaction (bool, optional): Whether to add atom as neighbor of itself(=self loop). Defaults to False.
            backend (Union[str, NeighborListBuilder], optional): The backend for building neighborlist.
        """
        neighborlist_builder_cls = resolve_neighborlist_builder(backend)
        neighborlist_builder: NeighborListBuilder = neighborlist_builder_cls(cutoff, self_interaction)
        center_idx, neigh_idx, edge_shift = neighborlist_builder.build(self)
        # edge index: [dst, src]
        edge_index = torch.stack([neigh_idx, center_idx], dim=0)
        self.edge_index = edge_index
        self.edge_shift = edge_shift

    def to_ase(self) -> Atoms:
        """Convert to Atoms object.

        Returns:
            Atoms: ASE Atoms object.
        """
        if self.cell.norm().abs() < 1e-6:
            pbc = False
        else:
            pbc = True
        atoms = Atoms(
            numbers=self.elems.detach().cpu().numpy(),
            positions=self.pos.detach().cpu().numpy(),
            cell=self.cell.detach().cpu().numpy()[0] if pbc else None,
            pbc=pbc,
        )
        if hasattr(self, "fixed_atoms"):
            atoms.constraints = FixAtoms(self.fixed_atoms.detach().cpu().numpy())
        energy = self.energy.detach().cpu().item() if "energy" in self else None
        forces = self.force.detach().cpu().numpy() if "force" in self else None
        if energy is not None or forces is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return atoms

    def volume(self) -> Tensor:
        """Return volume of the cell.

        Returns:
            Tensor: Volume of the cell.
        """
        return self.cell.squeeze().det()

    def compute_edge_vecs(self) -> "AtomsGraph":
        """Compute edge vectors from edge_index and edge_shift.

        Returns:
            AtomsGraph: self with ``
            edge_vec``.
        """

        if "edge_index" not in self:
            raise ValueError("Neighbor list is not built.")
        pos = self.pos
        batch = self.batch
        dst, src = self.edge_index
        edge_batch = batch[src]  # batch index for edges(neighbors)
        vec = pos[dst] - pos[src]
        cell = self.cell if "cell" in self else torch.zeros((batch.max() + 1, 3, 3)).to(vec.device)
        vec += torch.einsum("ni,nij->nj", self.edge_shift, cell[edge_batch])
        self.edge_vec = vec
        return self

    @staticmethod
    def resolve_cell(atoms: Atoms) -> Tensor:
        """Resolve cell as tensor from Atoms object with checking pbc.
        If pbc is False, return zeros.

        Args:
            atoms (Atoms): ASE Atoms object.

        Returns:
            Tensor: 1x3x3 tensor of lattice vectors. (1 is batch dimension)
        """
        # reject partial pbc
        if atoms.pbc.any() and not atoms.pbc.all():
            raise ValueError("AtomsGraph does not support partial pbc")
        pbc = atoms.pbc.all()
        if pbc:
            return torch.tensor(atoms.cell.array, dtype=_default_dtype).unsqueeze(0)
        # Return zeros when pbc is false
        return torch.zeros((1, 3, 3), dtype=_default_dtype)
