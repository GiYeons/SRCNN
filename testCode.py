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
from skimage.feature import canny
from skimage import io


bird = io.imread('/home/wstation/Set5/bird.bmp')
bird = rgb2ycbcr(bird)[:, :, 0]
edges = canny(bird, 4).astype(float)
edges = np.expand_dims(edges, axis=(0, 1))
edges = torch.tensor(edges)

edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=3//2)
edges = edges.numpy()[0, 0]

bird2 = imread('/home/wstation/Set5/bird.bmp')


fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(edges, cmap=plt.cm.gray)
ax.set_title('Canny detector')
ax.axis('off')
plt.show()