import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math as math
from skimage.io import imread, imshow
from matlab import imresize
from torchvision.utils import save_image

class Tconv_block(nn.Module):
    def __init__(self, scale, in_c, out_c, ker, r):
        super(Tconv_block, self).__init__()
        self.scale = scale
        self.ker = ker

        self.high_par = nn.ConvTranspose2d(
            in_channels=in_c, out_channels=out_c, kernel_size=ker, padding=ker//2, stride=scale, output_padding=scale-1)

        self.low_par1 = nn.Conv2d(
            in_channels=in_c, out_channels=in_c//r, kernel_size=1)
        self.low_par2 = nn.ConvTranspose2d(
            in_channels=in_c//r, out_channels=out_c, kernel_size=ker, padding=ker//2, stride=scale, output_padding=scale-1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        # ===========
        self.ones = nn.Parameter(data=torch.ones(size=(in_c, 1, 1, 1)).float(), requires_grad=False)
        # ===========

    def expand_hw(self, x):
        b, c, h, w = x.shape
        x = F.conv_transpose2d(x, self.ones[0:c], stride=(self.scale, self.scale), output_padding=self.scale-1, groups=c)
        return x
                
    def tconv_to_conv_par(self, par):
        par = torch.rot90(par, 2, [2, 3])
        par = par.transpose(0, 1)
        return par

    def eval_forward(self, x, mask_idx, inv_mask_idx):
        high_par = self.tconv_to_conv_par(self.high_par.weight)
        low_par1 = self.low_par1.weight
        low_par2 = self.tconv_to_conv_par(self.low_par2.weight)

        x = self.expand_hw(x)
        b, c, h, w = x.shape
        cout, cin, ker, ker = high_par.data.shape

        patches = F.pad(x, pad=(ker // 2, ker // 2, ker // 2, ker // 2))
        patches = patches.unfold(2, ker, 1).unfold(3, ker, 1)
        patches = patches.transpose(0, 1)
        patches = patches.contiguous().view(cin, -1, ker, ker)
        patches = patches.transpose(0, 1)

        patches_out = patches.new(b * h * w, cout, 1, 1)
        patches_out[mask_idx] = F.conv2d(patches[mask_idx], high_par)
        patches_out[inv_mask_idx] = F.conv2d(F.conv2d(patches[inv_mask_idx], low_par1), low_par2)
        patches = patches_out

        patches = patches.view(b, h * w, cout)
        patches = patches.transpose(2, 1)
        y = F.fold(patches, (h, w), (1, 1))

        return y

    def forward(self, x, mask, inv_mask, eval=False):
        if eval==True:
            return self.eval_forward(x, mask, inv_mask)

        high = self.high_par(x) * mask
        low = self.low_par1(x)
        low = self.low_par2(low) * inv_mask

        return high + low


class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, ker, r):
        super(Conv_block, self).__init__()

        self.high_par = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ker, padding=ker // 2)

        self.low_par1 = nn.Conv2d(in_channels=in_c, out_channels=out_c // r, kernel_size=ker, padding=ker // 2)
        self.low_par2 = nn.Conv2d(in_channels=out_c // r, out_channels=out_c, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

    def eval_forward(self, x, mask_idx, inv_mask_idx):
        b, c, h, w = x.shape
        high_par = self.high_par.weight
        low_par1 = self.low_par1.weight
        low_par2 = self.low_par2.weight
        cout, cin, ker, ker = high_par.data.shape

        patches = F.pad(x, pad=(ker//2, ker//2,ker//2, ker//2))
        patches = patches.unfold(2, ker, 1).unfold(3, ker, 1)
        patches = patches.transpose(0,1)
        patches = patches.contiguous().view(cin, -1, ker, ker)
        patches = patches.transpose(0,1)

        patches_out = patches.new(b * h * w, cout, 1, 1)
        patches_out[mask_idx] = F.conv2d(patches[mask_idx], high_par)
        patches_out[inv_mask_idx] = F.conv2d(F.conv2d(patches[inv_mask_idx], low_par1), low_par2)
        patches = patches_out

        patches = patches.view(b, h*w, cout)
        patches = patches.transpose(2,1)
        y = F.fold(patches, (h, w), (1, 1))

        return y

    def forward(self, x, mask, inv_mask, eval=False):
        if eval == True:
            return self.eval_forward(x, mask_idx=mask, inv_mask_idx=inv_mask)

        high = self.high_par(x) * mask
        low = self.low_par1(x) * inv_mask
        low = self.low_par2(low)

        return low + high


class Net(nn.Module):
    def __init__(self, scale, in_channel=1, num_filters1=16, num_filters2=32, r=4):
        super(Net, self).__init__()
        self.scale = scale

        self.first_part = Conv_block(in_c=in_channel, out_c=num_filters2, ker=5, r=r)
        self.reduction = Conv_block(in_c=num_filters2, out_c=num_filters1, ker=1, r=r)

        self.mid_part1 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)
        self.mid_part2 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)
        self.mid_part3 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)
        self.mid_part4 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)

        self.expansion = Conv_block(in_c=num_filters1, out_c=num_filters2, ker=1, r=r)
        self.last_part = Tconv_block(scale=scale, in_c=num_filters2, out_c=1, ker=9, r=r)

        self.relu = nn.ReLU(inplace=True)


    def create_mask(self, x, th, dilker, dilation=True):
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=3//2, count_include_pad=False)
        mask = torch.where(torch.abs(x-blur) >= th, 1, 0).float()

        if dilation==True:
            mask = self.dilate_mask(mask, dilker)
        inv_mask = torch.where(mask==1, 0, 1).float()

        # '''직접 마스크를 적용하는 임시 코드(사용후 반드시 삭제)'''
        # device = torch.device('cuda')
        # mask = imread('visualize/test.bmp')[:,:,0] / 255.
        # mask = np.expand_dims(mask, 0)
        # mask = np.expand_dims(mask, 0)
        # mask = torch.tensor(mask).to(device)
        # mask = mask.float()
        # mask = torch.where(mask==0, 0, 1)
        # inv_mask = torch.where(mask == 1, 0, 1).float()
        ''''''
        return mask, inv_mask

    def dilate_mask(self, mask, dilker):
        mask = F.max_pool2d(mask.float(), kernel_size=dilker, stride=1, padding=dilker//2)
        return mask

    def get_mask_index(self, mask):
        mask = torch.flatten(mask)
        inv_mask = torch.where(mask==1, 0, 1)

        mask_idx = torch.nonzero(mask)[:, 0]
        inv_mask_idx = torch.nonzero(inv_mask)[:, 0]

        return mask_idx, inv_mask_idx

    def upsample_mask(self, mask):
        mask = mask.repeat(1, self.scale**2, 1, 1)
        mask = F.pixel_shuffle(mask, self.scale)
        inv_mask = torch.where(mask==1, 0, 1)
        return mask, inv_mask


    def eval_forward(self, x, th=0.04, dilker=3, dilation=True):
        mask, inv_mask = self.create_mask(x, th, dilker, dilation)
        mask_idx, inv_mask_idx = self.get_mask_index(mask)


        x = self.relu(self.first_part(x,mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.reduction(x, mask_idx, inv_mask_idx, eval=True))

        x = self.relu(self.mid_part1(x, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.mid_part2(x, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.mid_part3(x, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.mid_part4(x, mask_idx, inv_mask_idx, eval=True))

        x = self.relu(self.expansion(x, mask_idx, inv_mask_idx, eval=True))

        mask, inv_mask = self.upsample_mask(mask)
        mask_idx, inv_mask_idx = self.get_mask_index(mask)
        y = self.last_part(x, mask_idx, inv_mask_idx, eval=True)

        return y

    def forward(self, x, th=0.04, dilker=3, dilation=True, eval=False):
        if eval == True:
            return self.eval_forward(x, th, dilker, dilation)

        mask, inv_mask = self.create_mask(x, th, dilker, dilation)

        orix = x

        x = self.relu(self.first_part(x, mask, inv_mask, eval=False))
        x = self.relu(self.reduction(x, mask, inv_mask, eval=False))

        x = self.relu(self.mid_part1(x, mask, inv_mask, eval=False))
        x = self.relu(self.mid_part2(x, mask, inv_mask, eval=False))
        x = self.relu(self.mid_part3(x, mask, inv_mask, eval=False))
        x = self.relu(self.mid_part4(x, mask, inv_mask, eval=False))

        x = self.relu(self.expansion(x, mask, inv_mask, eval=False))

        mask, inv_mask = self.upsample_mask(mask)
        y = self.last_part(x, mask, inv_mask, eval=False)



        return y

