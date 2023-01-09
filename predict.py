import os
from matlab import imresize, convertDouble2Byte as d2int
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr
import numpy as np
import torch
from utils import PSNR, rgb2y_uint8, SSIM
from torchvision.utils import save_image

# import the network we want to predict for
from model2 import Net

def val_psnr(model, th, dilker, dilation, val_path, scale):
    images = sorted(os.listdir(val_path))
    avg_psnr = 0
    avg_ssim = 0
    model.eval()
    with torch.no_grad():
        for image in images:
            img = imread(val_path + image)

            try:
                img = rgb2ycbcr(img)[:, :, 0:1]
            except:
                img = np.expand_dims(img, axis=-1)

            img = np.float64(img) / 255.
            height, width, channel = img.shape

            hr = img[0:height - (height % scale), 0: width - (width % scale), :]
            lr = imresize(hr, scalar_scale=1 / scale, method='bicubic')

            lr = np.moveaxis(lr, -1, 0)  # 텐서 계산을 위해 차원축 이동
            lr = np.expand_dims(lr, axis=0)  # 텐서 계산을 위해 차원 확장
            lr = torch.from_numpy(lr).float().to(device)

            out = model(lr)
            output = out.cuda().data.cpu()
            # save_image(output, f"{image}.png")
            output = output.numpy()

            hr = d2int(hr)
            output = d2int(output)

            output = output[0]
            output = np.moveaxis(output, 0, -1)

            avg_ssim +=SSIM(output, hr, scale)
            avg_psnr += PSNR(hr, output, boundary=scale)
            print(SSIM(output, hr, scale), "/", PSNR(hr, output, boundary=scale))


    # print(round(avg_ssim/len(images), 4), end=' ')
    return avg_psnr/len(images), avg_ssim/len(images)

device = torch.device('cuda')


sets = ['Set14/'] #, 'Set14/', 'BSDS100/', 'Urban100/', 'DIV2K_val/'''
dir_n = 'outputs/ASCNN/'
# paths = ['sp_02_dilk7', 'sp_04_dilk7', 'sp_06_dilk7', 'sp_08_dilk7']
# paths = ['sp_02_dilk3', 'sp_04_dilk3', 'sp_06_dilk3', 'sp_08_dilk3']
# paths = ['r8_th_01_dilk3', 'r8_th_04_dilk3', 'r8_th_07_dilk3', 'r8_th_10_dilk3']
# paths = ['th_01_dilk3', 'th_04_dilk3', 'th_07_dilk3', 'th_10_dilk3']
# paths = ['th_04_dilk3']
# paths = ['sp_02', 'sp_04', 'sp_06', 'sp_08']

# ths = [0.01, 0.025, 0.048, 0.088]
# ths = [0.0022, 0.0068, 0.016, 0.038]
# ths = [0.021, 0.05, 0.084, 0.13]
# ths = [0.03, 0.05, 0.07, 0.09]

paths = ['baseline']

# ths = [0.01, 0.04, 0.07, 0.10]
# ths = [0]
ths = [0.04]
dilation= True
dilker = 3

scale = 2
r = 16
model = Net(scale, r=r).float().to(device)
print('Number of Parameters:', sum(p.numel() for p in model.parameters()))


'''
the code below (from line 102) is to test for different models quickly at the same time
try to run without this code
just use: 
ckpnt = torch.load(path) # load weight path
model.load_state_dict(ckpnt) # load the weights to model
avg_psnr, avg_ssim = val_psnr(model, th, dilker, dilation, val_path, scale) # evaluate model with validation path
'''

for set in sets:
    for idx, path in enumerate(paths):
        path = dir_n + path + '.pth'
        ckpnt = torch.load(path)
        model.load_state_dict(ckpnt)
        val_path = '/home/wstation/' + set
        th = ths[idx]

        result1, result2 = val_psnr(model, th, dilker, dilation, val_path, scale)
        # print(round(result1, 5), end='/')
        # print(round(result2, 4), end='')
    print(end='\t')
