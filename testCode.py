import time

import torch
import matplotlib
import matplotlib.pyplot as plt
import h5py
import baseline
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import cv2
from matlab import *
from skimage.io import imread, imshow
from skimage.color import rgb2ycbcr
import torch.nn.modules as nm

from torch.utils.data import DataLoader, random_split
from datasets import TrainDataset
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from utils import *
from model import Net


# t = torch.tensor([[
#     [[1,2],
#     [3,4,]]]])
# print(t)
# t = t.repeat(1,4,1,1)
# print(t)
# t = F.pixel_shuffle(t, 2)
# print(t)


############## decision mask test #############
# plt.subplot(1,2,1)
# imshow(img, cmap='gray')
# img = np.moveaxis(img, 2, 0)
# img = torch.tensor(img).float() / 255.
# blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3//2, count_include_pad=False)
# mask = torch.where(torch.abs(img-blur) >= 0.04, 1, 0).float()
# mask = F.max_pool2d(mask.float(), kernel_size=3, padding=3//2)
#
# mask = np.moveaxis(mask.numpy(), 0, 2)
# plt.subplot(1,2,2)
# imshow(mask)
# plt.show()

# model = Net(2)
# outputs = model(img, 0.04, 3)
# print(outputs)


################ convTranspose2d test ####################
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.trans2d = nn.ConvTranspose2d(
#             in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2, bias=False)
#
#         self.trans2d.weight.data = nn.Parameter(torch.tensor([[[[0.1,0.2,0.3], [0.4,0.5,0.6]]]], dtype=torch.float32))
#
#     def forward(self, x):
#         y = self.trans2d(x)
#         return y
#
#
# t = torch.tensor([[[[1,2,3], [4,5,6]]]], dtype=torch.float32)
# model = Net()
# for param in model.parameters():
#     print(param)
#
# output = model(t)
# print(output)