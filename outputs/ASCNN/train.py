import time

import matplotlib.pyplot as plt
import h5py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from baseline import Net
import random
import numpy as np
import math
import os
import cv2
from matlab import *
from skimage.io import imread
from skimage.color import rgb2ycbcr
import gc

from datasets import Generator
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from utils import *


if __name__=="__main__":

    scales = [2]
    learnig_rate = 7e-4
    batch_size = 64
    epochs = 80
    iterations = 13000
    num_worker = 10

    r = 4
    th = 0.04
    dilker = 3
    dilation = False
    eval = False

    train_path = 'dataset/LR_HR/scale_x2/dataset_cpy1.h5'
    val_path = '/home/wstation/Set5/'
    output_path = '/outputs'

    device = torch.device('cuda')
    model = Net(scales[0]).float()
    model.to(device)
    print('Computation device: ', device)
    print('Number of Parameters:', sum(p.numel() for p in model.parameters()))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learnig_rate)

    data_gen = Generator(train_path, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=(epochs*iterations), eta_min=1e-10)

    best_psnr = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        #############train##############
        model.train()
        avg_loss = 0.
        start = time.time()

        for iter in range(iterations):
            lr, hr = next(data_gen)
            lr, hr = lr.to(device), hr.to(device)
            lr, hr = lr.float() / 255., hr.float() / 255.

            # prop
            optimizer.zero_grad()
            outs = model(lr, eval=eval)
            outputs = outs
            loss = criterion(outputs, hr)

            #backprop
            loss_limit = 10
            if float(loss) < loss_limit and float(loss) > -loss_limit:
                loss.backward()
                optimizer.step()

            #scheduler update
            learnig_rate = scheduler.get_last_lr()[0]
            scheduler.step()

            avg_loss += float(loss)
            progress_bar(iter, iterations, float(loss))

        avg_loss = avg_loss / iterations
        progress_bar(iterations, iterations, avg_loss)


        ############validation#############
        model.eval()
        with torch.no_grad():
            images = sorted(os.listdir(val_path))
            avg_psnr = 0.
            all_scales_avg_psnr = 0.

            for scale in scales:
                for image in images:
                    img = imread(val_path + image)
                    img = rgb2ycbcr(img)[:,:,0:1]
                    img = np.float64(img)/255.
                    height, width, channel = img.shape

                    hr = img[0:height - (height % scale), 0: width - (width % scale), :]
                    lr = imresize(hr, scalar_scale=1 / scale, method='bicubic')

                    lr = np.moveaxis(lr, -1, 0)  # 텐서 계산을 위해 차원축 이동
                    lr = np.expand_dims(lr, axis=0)   # 텐서 계산을 위해 차원 확장
                    lr = torch.from_numpy(lr).float().to(device)

                    out = model(lr, eval=eval)
                    output = out.cuda().data.cpu().numpy()

                    hr = convertDouble2Byte(hr)
                    output = convertDouble2Byte(output)

                    output = output[0]
                    output = np.moveaxis(output, 0, -1)

                    avg_psnr += PSNR(hr, output, boundary=scale)
                avg_psnr = avg_psnr / len(images)
                all_scales_avg_psnr += avg_psnr
                print(' - x' + str(scale) + '_val_psnr: {:.5f}'.format(avg_psnr), end=' ')
            all_scales_avg_psnr = all_scales_avg_psnr / len(scales)

            if best_psnr < all_scales_avg_psnr:
                best_weight = model.state_dict()
                best_psnr = all_scales_avg_psnr

        print('- lr: {:.7f}'.format(float(learnig_rate)), end=' ')
        end = time.time()
        print(f"{((end - start) / 60):.3f} minutes 소요됨")

    print(f"Max val PSNR: {best_psnr}")
    print('Saving model...')
    torch.save(best_weight, 'outputs/model.pth')

