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

def get_binary_mask(img, scale=2, th=0.04, max_pool=True):
    blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3//scale, count_include_pad=False)
    mask = torch.where(torch.abs(img-blur)>=th, 1, 0).float()
    if max_pool==True:
        mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=3//2)
    inv_mask = np.where(mask==1, 0, 1)

    return mask, inv_mask

def get_4path_mask(img, scale=2, th=[]):
    blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3 // scale, count_include_pad=False)
    loss = torch.abs(img - blur)
    mask_high = torch.where(loss >= th[0], 1, 0).float()
    mask_mid1 = torch.where(loss >= th[1], 1, 0)
    mask_mid2 = torch.where(loss >= th[2], 1, 0)
    mask_high = F.max_pool2d(mask_high.float(), kernel_size=3, stride=1, padding=3//2)
    mask_mid1 = F.max_pool2d(mask_mid1.float(), kernel_size=3, stride=1, padding=3//2)
    mask_mid2 = F.max_pool2d(mask_mid2.float(), kernel_size=3, stride=1, padding=3//2)
    mask_low = torch.where(mask_mid2 == 0, 1, 0)
    mask_mid2 = mask_mid2 - mask_mid1
    mask_mid1 = mask_mid1 - mask_high
    mask_high = mask_high.numpy()
    mask_mid1 = mask_mid1.numpy()
    mask_mid2 = mask_mid2.numpy()
    mask_low = mask_low.numpy()
    mask_high = np.where((mask_high==0) & (mask_mid1==1), 0.75, mask_high)
    mask_high = np.where((mask_high==0) & (mask_mid2==1), 0.5, mask_high)
    mask_high = np.where((mask_high==0) & (mask_low==1), 0.25, mask_high)

    return mask_high




def get_triple_mask_th(img, scale):
    blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3//scale, count_include_pad=False)
    frequency = torch.abs(img-blur)
    mask = torch.where(frequency>=0.04, 1, 0).double()
    mask = torch.where((mask!=1) & (frequency>=0.02), 0.5, mask).float()
    mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=3//2)

    return mask

def get_closing_mask(img, scale=2, th=0.05):
    blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3 // scale, count_include_pad=False)
    mask = torch.where(torch.abs(img - blur) >= th, 1, 0).float()
    mask_dil = F.max_pool2d(mask.float(), kernel_size=5, stride=1, padding=5 // 2)
    mask = -F.max_pool2d(-mask_dil, kernel_size=5, stride=1, padding=5 // 2)

    return mask, mask_dil


dir = '/home/wstation/TestSet/Set5/'
scale = 2
patch_size = 3

fname = 'bird.bmp'
img = imread(dir+fname)
img = normalize(conv_to_ycbcr(img))


# lr 생성
h, w, _ = img.shape
hr = img[0:h - (h % scale), 0: w - (w % scale), :]
lr = imresize(hr, scalar_scale=1 / scale, method='bicubic')
lr = np.moveaxis(lr, 2, 0)
lr = torch.tensor(lr)
_, h, w = lr.shape
lr = torch.round(lr, decimals=2) # 소수점 줄이기

# transposed convolution test



'''
# output 생성
mask_closing, mask_dilation = get_closing_mask(lr, th=0.02)
mask_baseline, _ = get_binary_mask(lr, th=0.04)
mask_4path = get_4path_mask(lr, th=[0.075, 0.035, 0.013])


sparsity1 = round(((h * w) - np.count_nonzero(mask_baseline)) /  (h * w), 2)
sparsity2 = round(((h * w) - np.count_nonzero(mask_closing)) /  (h * w), 2)



plt.subplot(121)
plt.title(f"baseline/{sparsity1}")
plt.imshow(mask_baseline[0], cmap='gray')
plt.subplot(122)
plt.title(f"dilation/{sparsity2}")
plt.imshow(mask_4path[0])
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
# plt.subplot(143)
# plt.title(f"erosion/{sparsity3}")
# plt.imshow(mask_redil[0], cmap='gray')
# plt.subplot(144)
# plt.title(f"dilation/{sparsity4}")
# plt.imshow(mask3[0], cmap='gray')
plt.show()

# save_image(mask_4path[0], f'outputs/4path.bmp')
# tm = mask_4path[0]
# tm = torch.Tensor(tm)
# save_image(tm, f'4path.bmp')
# hm = np.where(th_bi_mask==0, 0, 1)
# hm = torch.Tensor(hm)
# save_image(hm, f'thHighMask.bmp')
# diff = np.abs(tm-hm)
# save_image(diff, f'diff.bmp')
'''



# 라플라시안 테스트
"""
x = torch.tensor([[-1.,0.,1.],
                    [-2.,0.,2.],
                    [-1.,0.,1.]])
y = torch.tensor([[-1.,-2.,-1.],
                  [0.,0.,0.],
                  [1.,2.,1.]])
x = x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
y = y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

output = F.conv2d(lr.float(), x)
output = F.conv2d(output, y)
output = torch.where(torch.abs(output)>=0.8, 1, 0)
output = F.max_pool2d(output.float(), kernel_size=3, stride=1, padding=3//2)
output = output.numpy()[0]
print(output)
"""

##### fold unfold 예시
"""
# mask 생성
bmask, inv_bmask = get_binary_mask(lr, scale)
bmask_index = bmask.flatten()

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
high_mask_index = np.where(vars==-1, 1, 0)  # high mid low 에서의 high mask index
high_mask = high_mask_index.reshape(h, w)
triple_mask = np.where(vars>=0.002, 0.5, high_mask_index).reshape(h,w)   # visualizing을 위한 mask

triple_mask = np.expand_dims(triple_mask, 0)
triple_mask = F.max_pool2d(torch.tensor(triple_mask), kernel_size=3, stride=1, padding=3//2)
triple_mask = triple_mask.numpy()[0,:,:]

sparsity = round(((h * w) - np.count_nonzero(triple_mask)) /  (h * w), 2)


th_bi_mask = get_triple_mask_th(lr, scale)   # th를 조정한 baseline
sparsity_thm = round(((h * w) - np.count_nonzero(th_bi_mask)) /  (h * w), 2)
"""


