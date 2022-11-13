import math

import torch
import numpy as np
from math import log10, sqrt

def d2bytes(I): # 0~1사이 값을 받아 클리핑, 0~255로 리턴
    B = np.clip(I, 0.0, 1.0)    # min보다 작은 값을 min값으로, max보다 큰 값을 max값으로 바꿔줌
    B = 255.0 * B
    return np.around(B).astype(np.uint8)


def PSNR(label, outputs, boundary=0):
    # label = d2bytes(label.cpu().detach().numpy())
    # outputs = d2bytes(outputs.cpu().detach().numpy())
    label = d2bytes(label)
    outputs = d2bytes(outputs)

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