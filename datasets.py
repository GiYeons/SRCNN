import random

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

def Generator(path, batch_size=64, shuffle=True, num_workers=3):
    dataset = TrainDataset(path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    while(True):
        for lr_batch, hr_batch in data_loader:
            yield lr_batch, hr_batch


class TrainDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = h5py.File(self.path, 'r')
        self.x_set = self.dataset.get('x_set')
        self.y_set = self.dataset.get('y_set')

    def __len__(self):
        return len(self.x_set)

    def __getitem__(self, index):
        lr = self.x_set[str(index+1)][:]
        hr = self.y_set[str(index+1)][:]
        lr, hr = self.randomAugumentation(torch.from_numpy(lr), torch.from_numpy(hr))
        return lr, hr

    def randomAugumentation(self, lr, hr):
        angle = random.choice([0,1,2,3])
        if angle==0:
            pass
        else:
            lr = torch.rot90(lr, k=angle, dims=[1,2])
            hr = torch.rot90(hr, k=angle, dims=[1,2])

        flip = random.choice([0,1])

        if flip==0:
            pass
        else:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])

        return lr, hr

