import torch
from torch import nn
import torch.nn.functional as F
import math as math
from matlab import imresize

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, ker, r):
        super(Conv_block, self).__init__()

        self.high_par = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ker, padding=ker//2)

        self.low_par1 = nn.Conv2d(in_channels=in_c, out_channels=out_c//r, kernel_size=ker, padding=ker//2)
        self.low_par2 = nn.Conv2d(in_channels=out_c//r, out_channels=out_c, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

    def eval_forward(self, x, mask, inv_mask):
        pass

    def forward(self, x, mask, inv_mask, eval=False):
        if eval==True:
            return self.eval_forward(x, mask=mask, inv_mask=inv_mask)

        high = self.high_par(x) * mask
        low = self.low_par1(x) * inv_mask
        low = self.low_par2(low)

        return low + high


class Tconv_block(nn.Module):
    def __init__(self, scale, in_c, out_c, ker, r):
        super(Tconv_block, self).__init__()

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

    def eval_forward(self):
        pass

    def forward(self, x, mask, inv_mask, eval=False):
        if eval==True:
            return self.eval_forward()

        high = self.high_par(x) * mask
        low = self.low_par1(x)
        low = self.low_par2(low) * inv_mask

        return high + low


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


    def eval_forward(self, x):
        pass

    def forward(self, x, th=0.04, dilker=3, dilation=True, eval=False):
        if eval == True:
            self.eval_forward(x)

        mask, inv_mask = self.create_mask(x, th, dilker, dilation)


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

