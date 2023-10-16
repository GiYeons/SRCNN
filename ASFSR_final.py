import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math as math
from skimage.io import imread, imshow
from matlab import imresize
from torchvision.utils import save_image
torch.set_printoptions(sci_mode=False, precision=5)

class Tconv_block(nn.Module):
    def __init__(self, scale, in_c, out_c, ker, r):
        super(Tconv_block, self).__init__()
        self.scale = scale
        # sub-pixel convolution layer
        self.ker = 3
        self.out_c = int(out_c * (scale ** 2))

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=self.out_c, kernel_size=self.ker, padding=self.ker // 2),
            nn.PixelShuffle(scale),
        )

        # original code (transposed convolution layer)
        """
        self.ker = ker
        
        self.high_par = nn.ConvTranspose2d(
            in_channels=in_c, out_channels=out_c, kernel_size=ker, padding=ker//2, stride=scale, output_padding=scale-1)

        self.low_par1 = nn.Conv2d(
            in_channels=in_c, out_channels=in_c//r, kernel_size=1)
        self.low_par2 = nn.ConvTranspose2d(
            in_channels=in_c//r, out_channels=out_c, kernel_size=ker, padding=ker//2, stride=scale, output_padding=scale-1)
        """

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

    def forward(self, x, mask, inv_mask, eval=False):
        # original code
        """
        if eval==True:
            return self.eval_forward(x, mask, inv_mask)

        high = self.high_par(x) * mask
        low = self.low_par1(x)
        low = self.low_par2(low) * inv_mask
        """
        result = self.sub_pixel(x)

        return result


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

    def forward(self, x, mask, inv_mask, eval=False):
        high_bias = self.high_par.bias.data

        high = self.high_par(x) * mask
        low = self.low_par1(x) * inv_mask
        low = self.low_par2(low)

        return (low + high)


class Net(nn.Module):
    def __init__(self, scale, in_channel=1, num_filters1=16, num_filters2=32, r=4):
        super(Net, self).__init__()
        self.scale = scale
        self.num_filters1 = 16
        self.num_filters2 = 32

        self.first_part = Conv_block(in_c=in_channel, out_c=num_filters2, ker=5, r=r)
        self.reduction = Conv_block(in_c=num_filters2, out_c=num_filters1, ker=1, r=r)

        self.mid_part1 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)
        self.mid_part2 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)
        self.mid_part3 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)
        self.mid_part4 = Conv_block(in_c=num_filters1, out_c=num_filters1, ker=3, r=r)

        self.expansion = Conv_block(in_c=num_filters1, out_c=num_filters2, ker=1, r=r)
        self.last_part = Tconv_block(scale=scale, in_c=num_filters2, out_c=1, ker=9, r=r)

        self.relu = nn.ReLU(inplace=True)

        ##################################
        self.wts_nbit, self.wts_fbit = 8, 4
        self.biases_nbit, self.biases_ibit = 8, 4
        self.biases_fbit = self.biases_nbit - self.biases_ibit
        self.bit_shift, self.output_fbit = 0, 0
        self.act_nbit, self.act_fbit = 8, 4
        self.origin_wts = []
        self.quantized_wts = []


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

    #-----------------------quantization------------------------------
    # quantized된 weights를 모델의 weights에 대입하는 함수
    def quantize(self, scheme="uniform", wts_nbit=8, wts_fbit=4):
        self.prepare_q_weight(scheme, wts_nbit, wts_fbit)

        for i in range(len(self.all_layers) - 1):
            self.all_layers[i].high_par.weight.data = self.quantized_wts[i][0]
            self.all_layers[i].low_par1.weight.data = self.quantized_wts[i][1]
            self.all_layers[i].low_par2.weight.data = self.quantized_wts[i][2]
            self.all_layers[i].cuda()

    # model의 weight를 original weights로 되돌리는 함수
    def revert(self):
        for i in range(len(self.all_layers) - 1):
            self.all_layers[i].high_par.weight.data = self.origin_wts[i][0]
            self.all_layers[i].low_par1.weight.data = self.origin_wts[i][1]
            self.all_layers[i].low_par2.weight.data = self.origin_wts[i][2]
            self.all_layers[i].cuda()

    # quantized된 weights를 생성하는 함수
    def prepare_q_weight(self, scheme="uniform", wts_nbit=8, wts_fbit=4):
        self.wts_nbit = wts_nbit
        self.wts_fbit = wts_fbit

        # made for loof
        self.all_layers = nn.Sequential(
            self.first_part, self.reduction, self.mid_part1, self.mid_part2, self.mid_part3, self.mid_part4, self.expansion, self.last_part
        )
        #TConv에서 high, low weight를 없앴으므로 따로 처리해야 함
        for i in range(len(self.all_layers) - 1):
            high_wts = self.all_layers[i].high_par.weight.data
            low_wts1 = self.all_layers[i].low_par1.weight.data
            low_wts2 = self.all_layers[i].low_par2.weight.data

            # test (should be modified)
            wts_step = 2 ** -wts_fbit

            print('step: ', wts_step)
            if(scheme == "none"):
                pass

            elif(scheme == "uniform"):
                high_weight, _ = self.uniform_quantize(high_wts, wts_step, wts_nbit)
                low_weight1, _ = self.uniform_quantize(low_wts1, wts_step, wts_nbit)
                low_weight2, _ = self.uniform_quantize(low_wts2, wts_step, wts_nbit)

            elif(scheme == "scale_linear"):
                wts_nlevel = 2 ** wts_nbit
                output_channels = self.num_filters1     # 레이어에 따라 달리 적용해야 함
                output, w_bonus_scale_factor = self.scale_linear_quantize(high_wts, output_channels)

            self.origin_wts.append([high_wts, low_wts1, low_wts2])
            self.quantized_wts.append([high_weight, low_weight1, low_weight2])

            high_bias = self.all_layers[i].high_par.bias.data
            low_bias1 = self.all_layers[i].low_par1.bias.data
            low_bias2 = self.all_layers[i].low_par2.bias.data

            self.bit_shift, self.output_fbit = self.calculate_bit_shift(8, 4)
            print(self.bit_shift, self.output_fbit)


        # print(self.all_layers[0].high_par.weight.data)

    def uniform_quantize(self, wts, step, nbit):
        pos_end = 2 ** nbit - 1     # 255
        neg_end = -pos_end          # -255

        output = 2 * torch.round(wts / step + 0.5) - 1
        output = torch.clip(output, min=neg_end, max=pos_end)

        output_store = (output - 1) / 2

        return output, output_store

    def scale_linear_quantize(self, wts, n_out_channel, n_coff):
        nwts = torch.numel(wts)
        nwts_per_channel = nwts / n_out_channel

        scale_factor = torch.zeros(n_out_channel)

        # should be modified

    def calculate_bit_shift(self, input_ibit, input_fbit):
        scheme = "uniform"  # test
        activation = "float_relu"

        scales_nbit, scales_ibit = 8, 4
        biases_nbit, biases_ibit = 8, 4

        if(scheme == "none"):
            wts_fbit = 0
        elif(scheme == "uniform"):
            wts_fbit = self.wts_fbit + 1

        scales_fbit = scales_nbit - scales_ibit
        biases_fbit = biases_nbit - biases_ibit

        bit_shift = input_fbit + wts_fbit + scales_fbit - biases_fbit

        if(activation == "float_relu"):
            act_fbit = biases_fbit

        output_fbit = act_fbit
        input_fbit = act_fbit

        return bit_shift, output_fbit

    def relu_quantize(self, x, step, nbit, bias_shift):
        pos_end = torch.tensor(2 ** nbit - 1).to(x.device)

        activations = x;
        activations = torch.where(activations >= 0, activations / round(2 ** bias_shift * (step)), torch.zeros_like(activations))
        activations = torch.where(activations > pos_end, pos_end.float(), activations)

        return activations


    def forward(self, x, th=0.04, dilker=3, dilation=True, eval=False):

        mask, inv_mask = self.create_mask(x, th, dilker, dilation)

        orix = x

        x = self.relu_quantize(self.first_part(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        # insert activation quantization function?
        x = self.relu_quantize(self.reduction(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)

        x = self.relu_quantize(self.mid_part1(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        x = self.relu_quantize(self.mid_part2(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        x = self.relu_quantize(self.mid_part3(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        x = self.relu_quantize(self.mid_part4(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)

        x = self.relu_quantize(self.expansion(x, mask, inv_mask, eval=False), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        # x = self.relu(self.first_part(x, mask, inv_mask, eval=False))
        # # insert activation quantization function?
        # x = self.relu(self.reduction(x, mask, inv_mask, eval=False))
        #
        # x = self.relu(self.mid_part1(x, mask, inv_mask, eval=False))
        # x = self.relu(self.mid_part2(x, mask, inv_mask, eval=False))
        # x = self.relu(self.mid_part3(x, mask, inv_mask, eval=False))
        # x = self.relu(self.mid_part4(x, mask, inv_mask, eval=False))
        #
        # x = self.relu(self.expansion(x, mask, inv_mask, eval=False))

        mask, inv_mask = self.upsample_mask(mask)
        y = self.last_part(x, mask, inv_mask, eval=False)



        return y

