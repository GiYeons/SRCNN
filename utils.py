import math
import cv2
import torch
import numpy as np
from math import log10, sqrt

def d2bytes(I): # 0~1사이 값을 받아 클리핑, 0~255로 리턴
    B = np.clip(I, 0.0, 1.0)    # min보다 작은 값을 min값으로, max보다 큰 값을 max값으로 바꿔줌
    B = 255.0 * B
    return np.around(B).astype(np.uint8)

def rgb2y_uint8(rgb):
    y = (rgb[:, :, 0] * np.array([0.256789])) + (rgb[:, :, 1] * np.array([0.504129])) + (rgb[:, :, 2] * np.array([0.097906])) + 16
    y = np.clip(y, 0.0, 255)
    y =  np.around(y).astype(np.uint8)

    return y

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def SSIM(img1, img2, boundary):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    img1 = np.uint8(img1*255)
    img2 = np.uint8(img2 * 255)

    if boundary > 0:
        img1 = img1[boundary:-boundary, boundary:-boundary]
        img2 = img2[boundary:-boundary, boundary:-boundary]

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def PSNR(label, outputs, boundary=0):
    # label = d2bytes(label)
    # outputs = d2bytes(outputs)

    if boundary > 0:
        label = label[boundary:-boundary, boundary:-boundary]
        outputs = outputs[boundary:-boundary, boundary:-boundary]

    label = label.astype(np.float64)
    outputs = outputs.astype(np.float64)

    imdiff = label- outputs

    mse = np.mean(imdiff**2)
    rmse = sqrt(mse)
    psnr = 20 * log10(255/rmse)

    return psnr

def train_psnr(mse, max_pixel = 1.0):

    if(mse == 0):
        return 100
    try:
        psnr = 20 * log10(max_pixel / sqrt(mse))
    except:
        return 0

    return psnr

def progress_bar(batch, n_steps, loss):
    lenght = 30
    load = int((batch/n_steps) * lenght)
    dot = lenght-load

    print('\r{}/{}'.format(batch, n_steps), end=' ')
    print('[', end='')
    for i in range(1, load+1):
        if load == lenght and i==load: print('=', end='')
        elif i == load: print('>', end='')
        else: print('=', end='')
    for i in range(dot): print('.', end='')
    print(']', end=' ')

    print('- loss: {:.5f} - psnr: {:.4f}'.format(loss, train_psnr(loss)), end='')

def clock(seconds):
    if seconds < 60:
        return 0, seconds

    min = int(seconds / 60)
    sec = seconds % 60
    return min, sec

def hamming_distance(a, b): # 수정해서 사용할 것. if dst/총 픽셀수 < 0.25: # 해밍거리 25% 이내만 출력 ---⑨
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a !=b).sum()
    return distance
