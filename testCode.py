import time

import torch
import matplotlib
import matplotlib.pyplot as plt
import h5py
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
from baseline import Net
from skimage.feature import canny
from skimage import io
from predict import predict



img = io.imread('/home/wstation/Set5/bird.bmp')
img = rgb2ycbcr(img)[:128, :128, 0:1]
bird = img / 255.
bird = imresize(bird, scalar_scale=1 / 2, method='bicubic')
bird2 = np.expand_dims(np.moveaxis(bird, 2,0), axis=0)
bird2 = torch.tensor(bird2)
blur = F.avg_pool2d(bird2, kernel_size=3, stride=1, padding=3//2, count_include_pad=False)
blur = blur[0, 0].numpy()
bird = bird[:,:,0]
mask = np.abs(bird-blur)
mask2 = np.where(mask>0.04, 1, 0)
mask2 = F.max_pool1d(torch.tensor(mask2).float(), kernel_size=3, stride=1, padding=3//2)
mask2 = mask2.numpy()
mask3 = np.where(mask2==1, 0, 1)
mask3 = np.where((mask3==1) & (mask>0.002), 1, 0)


print(mask3.shape)



plt.subplot(141)
plt.imshow(blur, cmap='gray')
plt.title('original')
plt.axis('off')
plt.subplot(142)
plt.imshow(mask2, cmap='gray')
plt.title('high frequency')
plt.axis('off')
plt.subplot(143)
plt.imshow(mask3, cmap='gray')
plt.title('low frequency')
plt.axis('off')
plt.subplot(144)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# bird = io.imread('/home/wstation/Set5/bird.bmp')
# bird = rgb2ycbcr(bird)[:, :, 0]
# edges = canny(bird, 3).astype(float)
# edges = np.expand_dims(edges, axis=(0, 1))
# edges = torch.tensor(edges)
#
# # edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=3//2)
# edges = edges.numpy()[0, 0]
#
#
# bird2 = imread('/home/wstation/Set5/bird.bmp')
# bird2 = rgb2ycbcr(bird2)[:, :, 0:1] / 255.
# bird2 = np.expand_dims(np.moveaxis(bird2, 2,0), axis=0)
# bird2 = torch.tensor(bird2)
#
# blur = F.avg_pool2d(bird2, kernel_size=3, stride=1, padding=3//2, count_include_pad=False)
# mask = torch.where(torch.abs(bird2-blur) >= 0.03, 1, 0).float()
# # mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=3//2)
# mask = mask[0, 0]
#
#
# plt.subplot(121)
# plt.imshow(edges, cmap=plt.cm.gray)
# plt.title('Canny detector')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(mask, cmap=plt.cm.gray)
# plt.title('Thresholding')
# plt.axis('off')
# plt.show()