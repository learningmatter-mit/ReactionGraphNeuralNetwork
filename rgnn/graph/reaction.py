import torch
from ase import Atoms
from torch_geometric.typing import OptTensor, Tensor

from .atoms import AtomsGraph

_default_dtype = torch.get_default_dtype()


class ReactionGraph(AtomsGraph):
    def __init__(
        self,
        elems: Tensor = None,
        pos: Tensor = None,
        cell: OptTensor = None,
        edge_index: OptTensor = None,
        edge_shift: OptTensor = None,
        n_atoms_i: OptTensor = None,
        n_atoms_f: OptTensor = None,
        barrier: OptTensor = None,
        freq: OptTensor = None,
        energy_i: OptTensor = None,
        energy_f: OptTensor = None,
        delta_e: OptTensor = None,
        reaction_features: OptTensor = None,
        add_batch: bool = False,
        **kwargs,
    ):
        super().__init__(
            elems=elems,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            edge_shift=edge_shift,
            **kwargs,
        )
        self.barrier = barrier
        self.energy_i = energy_i
        self.energy_f = energy_f
        self.delta_e = delta_e
        self.freq = freq
        self.n_atoms_i = n_atoms_i
        self.n_atoms_f = n_atoms_f
        self.reaction_fatures = reaction_features
        if add_batch:
            self.batch = torch.zeros_like(elems, dtype=torch.long, device=pos.device)

    @classmethod
    def from_ase(
        cls,
        initial: Atoms,
        final: Atoms,
        barrier: OptTensor = None,
        freq: OptTensor = None,
        energy_i: OptTensor = None,
        energy_f: OptTensor = None,
        delta_e: OptTensor = None,
        neighborlist_cutoff: float | None = 5.5,
        device: str | torch.device | None = None,
        add_batch: bool = True,
    ):
        # Works only for the fixed volume
        device = "cpu" if device is None else device
        initial_atomsgraph = AtomsGraph.from_ase(
            initial, neighborlist_cutoff, read_properties=False, neighborlist_backend="ase", add_batch=add_batch
        )
        final_atomsgraph = AtomsGraph.from_ase(
            final, neighborlist_cutoff, read_properties=False, neighborlist_backend="ase", add_batch=add_batch
        )
        elems = torch.cat([initial_atomsgraph.elems, final_atomsgraph.elems], dim=0)
        pos = torch.cat([initial_atomsgraph.pos, final_atomsgraph.pos], dim=0)
        cell = initial_atomsgraph.cell
        edge_index_2 = final_atomsgraph.edge_index + initial_atomsgraph.elems.size(0)
        edge_index = torch.cat([initial_atomsgraph.edge_index, edge_index_2], dim=1)
        if initial_atomsgraph.edge_shift is not None and final_atomsgraph.edge_shift is not None:
            edge_shift = torch.cat([initial_atomsgraph.edge_shift, final_atomsgraph.edge_shift], dim=0)
        n_atoms_i = torch.tensor([len(initial)], dtype=torch.long, device=device)
        n_atoms_f = torch.tensor([len(final)], dtype=torch.long, device=device)
        # batch = torch.cat([initial_atomsgraph.batch, final_atomsgraph.batch], dim=0)
        # torch.as_tensor(energy_i, dtype=_default_dtype, device=device)
        if barrier is not None:
            barrier = torch.as_tensor([barrier], dtype=_default_dtype, device=device)
        if freq is not None:
            freq = torch.as_tensor([freq], dtype=_default_dtype, device=device)
        if energy_i is not None:
            energy_i = torch.as_tensor([energy_i], dtype=_default_dtype, device=device)
        if energy_f is not None:
            energy_f = torch.as_tensor([energy_f], dtype=_default_dtype, device=device)
        if delta_e is not None:
            delta_e = torch.as_tensor([delta_e], dtype=_default_dtype, device=device)
        elif energy_i is None or energy_f is None:
            delta_e = None
        else:
            delta_e = torch.sub(energy_f, energy_i)
        return cls(
            elems,
            pos,
            cell,
            edge_index,
            edge_shift,
            n_atoms_i,
            n_atoms_f,
            barrier,
            freq,
            energy_i,
            energy_f,
            delta_e,
            add_batch=add_batch,
        )

    def to_ase(self):

        if self.cell.norm().abs() < 1e-6:
            pbc = False
        else:
            pbc = True
        prev_atoms = Atoms(
            self.elems.detach().cpu().numpy()[: self.n_atoms_i],
            positions=self.pos.detach().cpu().numpy()[: self.n_atoms_i],
            cell=self.cell.detach().cpu().numpy()[0] if pbc else None,
            pbc=pbc,
        )

        future_atoms = Atoms(
            self.elems.detach().cpu().numpy()[self.n_atoms_i :],
            positions=self.pos.detach().cpu().numpy()[self.n_atoms_i :],
            cell=self.cell.detach().cpu().numpy()[0] if pbc else None,
            pbc=pbc,
        )
        return prev_atoms, future_atoms
