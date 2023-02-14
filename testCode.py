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
from skimage import io
from predict import predict
import collections
np.set_printoptions(precision=8, suppress=True)

def conv_to_ycbcr(img):    # return (h, w, 1)
    try:
        img = rgb2ycbcr(img)[:,:,0:1]
    except:
        img = np.expand_dims(img, -1)
    return img

def normalize(img):
    return img / 255.

def get_binary_mask(img, scale):
    blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3//scale, count_include_pad=False)
    mask = torch.where(torch.abs(img-blur)>=0.04, 1, 0).float()
    mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=3//2)
    inv_mask = np.where(mask==1, 0, 1)

    return mask, inv_mask

def get_binary_mask_th(img, scale):
    blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3//scale, count_include_pad=False)
    frequency = torch.abs(img-blur)
    mask = torch.where(frequency>=0.04, 1, 0).double()
    mask = torch.where((mask!=1) & (frequency>=0.02), 0.5, mask).float()
    mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=3//2)

    return mask




dir = '/home/wstation/TestSet/Set5/'
scale = 2
patch_size = 3

fname = 'bird.bmp'
img = imread(dir+fname)
img = normalize(conv_to_ycbcr(img))

# lr 생성
lr = imresize(img, scalar_scale=1 / scale, method='bicubic')
lr = np.moveaxis(lr, 2, 0)
lr = torch.tensor(lr)

# mask 생성
bmask, inv_bmask = get_binary_mask(lr, scale)
bmask_index = bmask.flatten()

# lr 및 mask 3x3 patch로 나누기
_, h, w = lr.shape

lr_simple = torch.round(lr, decimals=2)

patches_img = F.pad(lr_simple, pad=(patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2))
patches_img = patches_img.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
patches_img = patches_img.contiguous().view(-1, patch_size, patch_size)

patches_mask = F.pad(bmask, pad=(patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2))
patches_mask = patches_mask.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
patches_mask = patches_mask.contiguous().view(-1, patch_size, patch_size)

patches_img = np.where(patches_mask==1, 0, patches_img) # low pixel은 1, high pixel은 0

# 분산 구하기
vars = np.full(len(patches_img), -1, dtype='float32')    # high pixel의 인덱스에는 -1
for i, patch in enumerate(patches_img):
    patch_nonzero = patch[np.nonzero(patch)]
    if bmask_index[i]==0:
        vars[i] = np.var(patch_nonzero)

# create mask
high_mask_index = np.where((vars==-1) |(vars>=0.001), 1, 0)  # high mid low 에서의 high mask index
high_mask = high_mask_index.reshape(h, w)
triple_mask = np.where(vars>=0.0004, 0.5, high_mask_index).reshape(h,w)
sparsity = round(((h * w) - np.count_nonzero(triple_mask)) /  (h * w), 2)


th_bi_mask = get_binary_mask_th(lr,scale)   # th를 조정한 baseline
sparsity_thm = round(((h * w) - np.count_nonzero(th_bi_mask)) /  (h * w), 2)

plt.subplot(131)
plt.title(f"baseline")
plt.imshow(bmask[0])
plt.subplot(132)
plt.title(f"my try/sparsity:{sparsity}")
plt.imshow(triple_mask)
plt.subplot(133)
plt.title(f"baseline(th low ver)/sparsity:{sparsity_thm}")
plt.imshow(th_bi_mask[0])
plt.show()







##### 이미지 분류 및 hf ratio 비교
"""
def get_hf_ratio(img, scale):   # 고주파 영역/전체 이미지 픽셀 비율
    mask, inv_mask = get_mask(img, scale)
    h, w, c = mask.shape
    ratio = (np.count_nonzero(mask) / (h * w) * 100) / 100
    return ratio

def get_lf_var(img, scale):
    mask, inv_mask = get_binary_mask(img, scale)
    lr = imresize(img, scalar_scale=1 / scale, method='bicubic')
    lr = np.where(lr*mask!=0, lr, 0)
    lf = lr[lr.nonzero()]
    return np.var(lf)

def get_binary_mask(img, scale):
    lr = imresize(img, scalar_scale=1 / scale, method='bicubic')
    lr = np.moveaxis(lr, 2, 0)
    lr = torch.tensor(lr)
    blur = F.avg_pool2d(lr, kernel_size=3, stride=1, padding=3//scale, count_include_pad=False)
    mask = torch.where(torch.abs(lr-blur)>=0.04, 1, 0).float()
    mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=3//2)
    mask = np.moveaxis(mask.numpy(), 0, 2)
    inv_mask = np.where(mask==1, 0, 1)
#---------------------- main -------------------------
bads = []
goods = []
for name in fnames:
    img = io.imread(dir+name)

    img = conv_to_ycbcr(img)
    img = normalize(img)
    h, w, c = img.shape
    img[:h - (h%scale), :w - (w%scale), :]
    #분류
    if name in ['barbara.png', 'face.png', 'flowers.png', 'foreman.png', 'monarch.png', 'ppt3.png']:
        bads.append(img)
    else:
        goods.append(img)

goods_ratio = 0
bads_ratio = 0
for i, img in enumerate(goods):
    hfr = get_hf_ratio(img, scale)
    print(f'goods{i} hf ratio: {hfr:.2f}/ lf var: {get_lf_var(img, scale):.2f}')
    goods_ratio += hfr

print('------------------')
for i, img in enumerate(bads):
    hfr = get_hf_ratio(img, scale)
    print(f'bads{i} hf ratio: {hfr:.2f}/ lf var: {get_lf_var(img, scale):.2f}')
    bads_ratio += hfr

print(f'avg hf ratio(goods): {goods_ratio / len(goods):.2f}')
print(f'avg hf ratio(bads): {bads_ratio / len(bads):.2f}')

"""


