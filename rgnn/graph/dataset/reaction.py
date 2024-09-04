import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from rgnn.graph.reaction import ReactionGraph


class ReactionDataset(TorchDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def make_dataset(traj_reaction, traj_product, E_r, E_p, Ea, freq, **kwargs):
    """
    Create a ReactionDataset from reaction and product trajectories, energies, and frequencies.

    Parameters:
    traj_reaction (list): List of reaction trajectories (ASE atoms objects).
    traj_product (list): List of product trajectories (ASE atoms objects).
    E_r (list): Energies of reactants.
    E_p (list): Energies of products.
    Ea (list): Activation energies (energy barriers).
    freq (list): Frequencies associated with the trajectories.
    kwargs (dict): Additional attributes to add to the dataset. Each key should be a tuple where the first 
                   element is a list of values and the second element is the dtype.

    Returns:
    ReactionDataset: A dataset containing reaction and product data.
    """
    dataset_list = []
    for i in tqdm(range(len(traj_reaction))):
        data = ReactionGraph.from_ase(traj_reaction[i],
                                      traj_product[i],
                                      barrier=Ea[i],
                                      freq=freq[i],
                                      energy_i=E_r[i],
                                      energy_f=E_p[i])
        if kwargs is not None:
            for key, (values, dtype) in kwargs.items():
                setattr(data, key, torch.tensor([values[i]], dtype=dtype))
        dataset_list.append(data)

    dataset = ReactionDataset(dataset_list)

    return dataset
