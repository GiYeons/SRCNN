import time

import matplotlib.pyplot as plt
import h5py
import model
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import cv2
from matlab import *
from skimage.io import imread
from skimage.color import rgb2ycbcr

from torch.utils.data import DataLoader, random_split
from datasets import TrainDataset
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from utils import *


if __name__=="__main__":

    scale = 2
    lr = 1e-4
    batch_size = 32
    epochs = 80
    border_cut = 32//2//2
    num_workers = 10

    train_path = 'dataset/LR_HR/scale_x2/dataset_cpy1.h5'
    val_path = '/home/wstation/Set5/'
    output_path = '/outputs'

    img_rows, img_cols = 32, 32
    out_rows, out_cols = 64, 64

    # 트레이닝 데이터셋 준비
    dataset = TrainDataset(train_path)
    dataset_size = len(dataset)  # 896573 members
    train_size = 50000
    not_used_size = dataset_size-train_size     # 데이터가 너무 크니까 일부만 사용하려고 분리하는것
    # print(train_size+not_used_size)
    train_data, not_used_data = random_split(dataset, [train_size, not_used_size])  # dataset을 split

    # 데이터로더
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device('cuda:0')
    print('Computation device: ', device)
    model = model.SRCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1}
    ], lr=lr)


    # train
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        ############# train ###############
        model.train()
        running_loss = 0.0
        running_psnr = 0.0

        for bi, data in enumerate(tqdm(train_dataloader)):
            image_data = data[0].to(device)  # LR
            label = data[1].to(device)  # HR

            # zero grad(backward시에 미분한 값들이 누적되어 문제가 생기므로 초기화해야 함)
            optimizer.zero_grad()
            outputs = model(image_data, scale, border_cut)
            label = label[:, :, border_cut:-border_cut, border_cut:-border_cut]
            loss = criterion(outputs, label)

            # backpropagation
            loss.backward()  # 손실의 변화도를 갖고 있는 텐서
            # update the parameters
            optimizer.step()

            label = label.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            # 각 배치의 loss 누적하기
            running_loss += loss.item()  # batch loss
            batch_psnr = PSNR(label, outputs)
            running_psnr += batch_psnr

        train_epoch_loss = running_loss / len(train_dataloader.dataset)  # 데이터셋 개수만큼 나누기
        train_epoch_psnr = running_psnr / (int(len(train_dataloader.dataset) / train_dataloader.batch_size))  # 배치 개수만큼 나누기
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)


        ############ validation ############
        model.eval()  # training time과 다르게 작동하도록 해주는 스위치
        running_loss = 0.0
        running_psnr = 0.0

        images = sorted(os.listdir(val_path))

        with torch.no_grad():
            # set5 전처리 및 연산
            for image in images:
                # 전처리
                img = imread(val_path + image)
                img = rgb2ycbcr(img)[:, :, 0:1]
                height, width, channel = img.shape

                label = img[0:height - (height % scale), 0: width - (width % scale), :]
                lr = imresize(label, scalar_scale=1 / scale, method='bicubic')

                lr = np.moveaxis(lr, -1, 0) # 텐서 계산을 위해 차원축 이동
                lr = np.expand_dims(lr, axis=0)   # 텐서 계산을 위해 차원 확장
                label = np.moveaxis(label, -1, 0)
                label = np.expand_dims(label, axis=0)

                label = torch.tensor(label, dtype=torch.float32) / 255.
                lr = torch.tensor(lr, dtype=torch.float32) / 255.

                # 연산
                image_data = lr.to(device)  # LR
                label = label.to(device)  # LR
                outputs = model(image_data, scale, border_cut)

                label = label[:, :, border_cut:-border_cut, border_cut:-border_cut] #LR과 비교를 위해 크롭
                loss = criterion(outputs, label)

                save_image(outputs, f"outputs/val_sr{epoch}.png")

                label = label.cpu().detach().numpy()
                label = np.squeeze(label, 0)
                label = np.moveaxis(label, 0, -1)

                outputs = outputs.cpu().detach().numpy()
                outputs = np.squeeze(outputs, 0)
                outputs = np.moveaxis(outputs, 0, -1)

                running_loss += loss.item()
                running_psnr += PSNR(label, outputs) # 각 image의 pnsr 누적

        val_epoch_loss = running_loss / len(images)
        val_epoch_psnr = running_psnr / len(images)

        if(epoch>1):
            if (val_epoch_psnr > max(val_psnr)):
                best_weight = model.state_dict()

        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")


    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

    # save graphical plots and the model
    # loss
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')
    plt.show()

    # psnr
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='orange', label='train PSNR dB')
    plt.plot(val_psnr, color='red', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('outputs/psnr.png')
    plt.show()

    print(f"Max val PSNR: {max(val_psnr)}")
    print('Saving model...')
    torch.save(best_weight, 'outputs/model.pth')
