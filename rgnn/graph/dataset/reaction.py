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


def make_dataset(traj_reaction, traj_product, E_r, E_p, E_s, freq):
    dataset_list = []
    for i in tqdm(range(len(traj_reaction))):
        data = ReactionGraph.from_ase(traj_reaction[i],
                                      traj_product[i],
                                      barrier=E_s[i],
                                      freq=freq[i],
                                      energy_i=E_r[i],
                                      energy_f=E_p[i])
        dataset_list.append(data)

    dataset = ReactionDataset(dataset_list)

    return dataset
