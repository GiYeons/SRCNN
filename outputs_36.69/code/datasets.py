import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = None
        with h5py.File(self.path, 'r') as file:
            self.dataset_len = len(file['x_set'])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        self.dataset = h5py.File(self.path, 'r')
        self.lr = self.dataset['x_set'][str(index+1)]
        self.label = self.dataset['y_set'][str(index+1)]
        return (
            torch.tensor(np.array(self.lr), dtype=torch.float32) / 255.0,
            torch.tensor(np.array(self.label), dtype=torch.float32) / 255.0
        )
