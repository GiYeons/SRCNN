import os

import matplotlib.pyplot as plt

from matlab import imresize, convertDouble2Byte as d2int
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr
import numpy as np
import torch
from utils import PSNR, rgb2y_uint8, SSIM, PSNR_specific
from torchvision.utils import save_image
from torchsummary import summary as summary
from collections import OrderedDict

# import the network we want to predict for
from ASFSR_quantize_layerwise import Net

testWord = 'bird'   # line22, 134

def val_psnr(model, th, dilker, dilation, val_path, scale, boundary, psnrs):
    images = sorted(os.listdir(val_path))
    # images = [s for s in images if testWord in s] #############
    avg_psnr = 0
    avg_ssim = 0
    model.eval()
    with torch.no_grad():
        for image in images:
            print(val_path + image)
            img = imread(val_path + image)

            try:
                img = rgb2ycbcr(img)[:, :, 0:1]
            except:
                img = np.expand_dims(img, axis=-1)

            ####### image quantization ############
            img = np.clip(img, 0, 255)
            img = np.around(img).astype(np.uint8)
            #######################################
            height, width, channel = img.shape

            hr = img[0:height - (height % scale), 0: width - (width % scale), :]
            lr = imresize(hr, scalar_scale=1 / scale, method='bicubic')

            lr = np.moveaxis(lr, -1, 0)  # 텐서 계산을 위해 차원축 이동
            lr = np.expand_dims(lr, axis=0)  # 텐서 계산을 위해 차원 확장
            lr = torch.from_numpy(lr).float().to(device)

            out = model(lr, th=th)
            output = out.cuda().data.cpu()
            output = output.numpy()

            output = np.clip(output, 0, 255)
            output = np.around(output).astype(np.uint8)

            output = output[0]
            output = np.moveaxis(output, 0, -1)

            imsave(f"outputs/{image.replace('.bmp', '')}.png", output)

            avg_ssim +=SSIM(output, hr, boundary)
            avg_psnr += PSNR(hr, output, boundary=boundary)
            print(SSIM(output, hr, scale), "/", PSNR(hr, output, boundary=boundary))
            psnrs.append(PSNR(hr, output, boundary=boundary))

    # print(round(avg_ssim/len(images), 4), end=' ')
    return avg_psnr/len(images), avg_ssim/len(images)


def predict(datasets, model_paths, r=[4], th=[0.04]):


    sets = datasets #['set5/'] #, 'Set14/', 'BSDS100/', 'Urban100/', 'DIV2K_val/'''
    dir_n = 'outputs/ASCNN/'
    # paths = ['sp_02_dilk7', 'sp_04_dilk7', 'sp_06_dilk7', 'sp_08_dilk7']
    # paths = ['sp_02', 'sp_04', 'sp_06', 'sp_08']

    # ths = [0.01, 0.025, 0.048, 0.088]

    paths = model_paths # ['baseline']

    th = th
    dilation= True
    dilker = 3

    scale = 2
    r = r
    boundary = scale
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
    psnrs = []
    for set in sets:
        for idx, path in enumerate(paths):
            path = dir_n + path + '.pth'

            print(path)
            ckpnt = torch.load(path)
            model.load_state_dict(ckpnt, strict=False)
            val_path = '/home/dataset/' + set + "/HR/"

            #--------------------quantization-------------------------
            ## bias initialization
            ## bias를 반영하기 전에 bit_shift를 연산하기 위해 forward에서 manually하게 bias연산을 작성해야 함
            ## 따라서 bias를 수동으로 초기화한다
            ## chpnt 에서 **_part.**_par.bias 이름으로 된 것만 뽑아옴
            biases = OrderedDict()
            for key, value in ckpnt.items():
                if(key[-4:] == "bias"):
                    biases[key] = value

            model.init_bias(biases)

            ## weights quantization
            #quantization_settings: [(scheme, (wts_nbit, wts_fbit), (biases_nbit, biases_fbit), (act_nbit, act_fbit)), ...]
            quantization_settings = [
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), (16, 8)),
                    ("uniform", (16, 8), (16, 8), ( 0, 0)),
                ]

            model.assign_quantization_settings(quantization_settings)
            model.quantize()
            # for name, param in model.named_parameters():
                # print("파라미터 이름:", name, "파라미터", param)



            result1, result2 = val_psnr(model, th, dilker, dilation, val_path, scale, boundary, psnrs)
            print('avg PSNR:', round(result1, 5), end='/')
            print('avg SSIM:', round(result2, 4))

    return psnrs


if __name__=="__main__":
    device = torch.device('cuda')

    set = 'Set5/'
    psnrs = predict([set], ['final'], r=4, th=0.04)

    # images = sorted(os.listdir('/home/wstation/TestSet/'+set))
    # images = [s for s in images if testWord in s]

