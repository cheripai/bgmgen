import torch
from torch.utils.data import Dataset


class PairedArrayLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y 

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

    def __len__(self):
        return self.x.shape[0]
