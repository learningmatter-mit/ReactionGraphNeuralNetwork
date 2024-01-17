import ase.data
import numpy as np
import torch

# from aml.common.utils import canocialize_species
from rgnn.common import keys as K
from rgnn.common.typing import DataDict, Tensor


def canocialize_species(species: Tensor | list[int] | list[str]) -> Tensor:
    """Convert species to a tensor of atomic numbers.

    Args:
        species (Tensor | list[int] | list[str]): The species.
            One of the following:
                - A tensor of atomic numbers.
                - A list of atomic numbers.
                - A list of atomic symbols.

    Returns:
        Tensor: The species as a tensor of atomic numbers.
    """
    if isinstance(species[0], str):
        species = [ase.data.atomic_numbers[s] for s in species]
        species = torch.as_tensor(species, dtype=torch.long)
    elif isinstance(species, list) and isinstance(species[0], int):
        species = torch.as_tensor(species, dtype=torch.long)
    elif isinstance(species, np.ndarray):
        species = torch.as_tensor(species, dtype=torch.long)
    return species


class GlobalScaleShift(torch.nn.Module):
    """Scale and shift the energy.
    Caution: mean value is for per atom energy, not per molecule energy.

    Args:
        mean: mean value of energy.
        std: standard deviation of energy.
        key: key of energy in data dictionary.
    """

    def __init__(self, mean=0.0, std=1.0, key=K.energy):
        super().__init__()
        self.key = key
        self.register_buffer("scale", torch.tensor(std, dtype=torch.float))
        self.register_buffer("shift", torch.tensor(mean, dtype=torch.float))

    def forward(self, data: DataDict, energy: Tensor) -> Tensor:
        energy = energy * self.scale + self.shift * data[K.n_atoms]
        return energy


class PerSpeciesScaleShift(torch.nn.Module):

    def __init__(
        self,
        species,
        key=K.atomic_energy,
        initial_scales: dict[str, float] | None = None,
        initial_shifts: dict[str, float] | None = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.species = canocialize_species(species).sort()[0]
        self._trainable = trainable
        elem_lookup = torch.zeros(100, dtype=torch.long)
        elem_lookup[self.species] = torch.arange(len(self.species))
        self.register_buffer("elem_lookup", elem_lookup)
        self.key = key
        # Per-element scale and shifts
        self.scales = torch.nn.Parameter(torch.ones(len(self.species)),
                                         requires_grad=self.trainable)
        self.shifts = torch.nn.Parameter(torch.zeros(len(self.species)),
                                         requires_grad=self.trainable)
        if initial_scales is not None:
            scales = []
            for atomic_num in self.species:
                symbol = ase.data.chemical_symbols[atomic_num]
                scales.append(initial_scales[symbol])
            self.scales.data = torch.as_tensor(scales, dtype=torch.float32)
        if initial_shifts is not None:
            shifts = []
            for atomic_num in self.species:
                symbol = ase.data.chemical_symbols[atomic_num]
                shifts.append(initial_shifts[symbol])
            self.shifts.data = torch.as_tensor(shifts, dtype=torch.float32)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self._trainable = value
        self.scales.requires_grad_(value)
        self.shifts.requires_grad_(value)

    def forward(self, data: DataDict, atomic_energy: Tensor) -> Tensor:
        species = data[K.elems]
        idx = self.elem_lookup[species]
        atomic_energy = atomic_energy * self.scales[idx] + self.shifts[idx]
        return atomic_energy


class ScaleShift(torch.nn.Module):
    r"""Scale and shift layer for standardization.
    .. math::
       y = x \times \sigma + \mu
    Args:
        means (dict): dictionary of mean values
        stddev (dict): dictionary of standard deviations
    """

    def __init__(self, means=None, stddevs=None):
        super().__init__()
        print(means, stddevs)
        means = means if (means is not None) else {}
        stddevs = stddevs if (stddevs is not None) else {}
        self.means = means
        self.stddevs = stddevs

    def forward(self, key, inp):
        """Compute layer output.
        Args:
            inp (torch.Tensor): input data.
        Returns:
            torch.Tensor: layer output.
        """

        stddev = self.stddevs.get(key, 1.0)
        stddev = stddev.to(inp.device)
        mean = self.means.get(key, 0.0)
        mean = mean.to(inp.device)
        out = inp * stddev + mean

        return out