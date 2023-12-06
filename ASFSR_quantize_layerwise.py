import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math as math
from skimage.io import imread, imshow
from matlab import imresize
from torchvision.utils import save_image
torch.set_printoptions(sci_mode=False, precision=5)

#nghiant: moving this outside of class
def uniform_quantize(x, step, nbit):
    pos_end = 2 ** nbit - 1     # 255
    neg_end = -pos_end          # -255

    output = 2 * torch.round(x / step + 0.5) - 1
    output = torch.clip(output, min=neg_end, max=pos_end)

    return output

def quantize_and_constrain(x, nbit, fbit, sign=True):
    qfactor = 2 ** fbit
    nfactor = 2 ** nbit

    if(sign):
        maxv = nfactor / 2 - 1
        minv = maxv - nfactor + 1
    else:
        maxv = nfactor - 1
        minv = 0

    output = torch.round(x * qfactor)
    output = torch.clip(output, min=minv, max=maxv)

    return output

def relu_quantize(x, step, nbit, bias_shift):
    pos_end = torch.tensor(2 ** nbit - 1).to(x.device)

    activations = x;
    activations = torch.where(activations >= 0, activations / round(2 ** bias_shift * (step)), torch.zeros_like(activations))
    activations = torch.where(activations > pos_end, pos_end.float(), activations)

    return activations
#nghiant_end

class Tconv_block(nn.Module):
    def __init__(self, scale, in_c, out_c, ker, r):
        super(Tconv_block, self).__init__()
        self.scale = scale
        # sub-pixel convolution layer
        self.ker = 3
        self.out_c = int(out_c * (scale ** 2))

        self.sub_pixel = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=self.out_c, kernel_size=self.ker, padding=self.ker // 2, bias=False),
                                nn.PixelShuffle(scale))

        # custom bias
        self.use_quantization = False

        self.bias = None

        self.origin_wts = None
        self.quantized_wts = None
        self.origin_biases = None
        self.quantized_biases = None

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                # nn.init.zeros_(m.bias.data)

        # ===========
        self.ones = nn.Parameter(data=torch.ones(size=(in_c, 1, 1, 1)).float(), requires_grad=False)
        # ===========

    def assign_quantization_settings(self, settings):
        self.scheme, wts_bits, biases_bits, act_bits = settings
        self.wts_nbit, self.wts_fbit = wts_bits
        self.biases_nbit, self.biases_fbit = biases_bits
        self.act_nbit, self.act_fbit = act_bits

        origin_wts = self.sub_pixel[0].weight.data

        if self.scheme == "uniform":
            quantized_wts = uniform_quantize(origin_wts, 2 ** -self.wts_fbit, self.wts_nbit)
        # elif self.scheme == "other": #extend other schemes here
        #     pass

        self.origin_wts = origin_wts
        self.quantized_wts = quantized_wts

        ## biases
        origin_biases = self.bias
        quantized_biases = quantize_and_constrain(origin_biases, self.biases_nbit, self.biases_fbit)

        self.origin_biases = origin_biases
        self.quantized_biases = quantized_biases

    def forward(self, x, mask, inv_mask, input_fbit):
        # original code
        extra_bit = 0
        if self.scheme == "uniform":
            extra_bit = 1

        bit_shift = input_fbit + self.wts_fbit + extra_bit - self.biases_fbit
        x = torch.floor(self.sub_pixel[0](x) / 2 ** bit_shift) + self.bias
        result = self.sub_pixel[1](x)

        return result

    def quantize(self):
        self.use_quantization = True
        self.sub_pixel[0].weight.data = self.quantized_wts
        self.bias = self.quantized_biases

    def revert(self):
        self.use_quantization = False
        self.sub_pixel[0].weight.data = self.origin_wts
        self.bias = self.origin_biases


class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, ker, r):
        super(Conv_block, self).__init__()

        self.high_par = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ker, padding=ker // 2, bias=False)

        self.low_par1 = nn.Conv2d(in_channels=in_c, out_channels=out_c // r, kernel_size=ker, padding=ker // 2, bias=False)
        self.low_par2 = nn.Conv2d(in_channels=out_c // r, out_channels=out_c, kernel_size=1, bias=False)

        # custom bias
        self.high_bias = None
        self.low1_bias = None
        self.low2_bias = None

        self.origin_wts = []
        self.quantized_wts = []
        self.origin_biases = []
        self.quantized_biases = []

        self.use_quantization = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))

    def assign_quantization_settings(self, settings):
        self.scheme, wts_bits, biases_bits, act_bits = settings
        self.wts_nbit, self.wts_fbit = wts_bits
        self.biases_nbit, self.biases_fbit = biases_bits
        self.act_nbit, self.act_fbit = act_bits

        high_wts = self.high_par.weight.data
        low_wts1 = self.low_par1.weight.data
        low_wts2 = self.low_par2.weight.data

        # print(torch.max(high_wts), torch.min(high_wts))
        # print(torch.max(low_wts1), torch.min(low_wts1))
        # print(torch.max(low_wts2), torch.min(low_wts2))

        if self.scheme == "uniform":
            high_weight = uniform_quantize(high_wts, 2 ** -self.wts_fbit, self.wts_nbit)
            low_weight1 = uniform_quantize(low_wts1, 2 ** -self.wts_fbit, self.wts_nbit)
            low_weight2 = uniform_quantize(low_wts2, 2 ** -self.wts_fbit, self.wts_nbit)
        # elif self.scheme == "other": #extend other schemes here
        #     pass

        self.origin_wts = [high_wts, low_wts1, low_wts2]
        self.quantized_wts = [high_weight, low_weight1, low_weight2]

        # biases
        high_b = self.high_bias
        low_b1 = self.low1_bias
        low_b2 = self.low2_bias

        high_biases = quantize_and_constrain(high_b, self.biases_nbit, self.biases_fbit)
        low_biases1 = quantize_and_constrain(low_b1, self.biases_nbit, self.biases_fbit)
        low_biases2 = quantize_and_constrain(low_b2, self.biases_nbit, self.biases_fbit)

        self.origin_biases = [high_b, low_b1, low_b2]
        self.quantized_biases = [high_biases, low_biases1, low_biases2]

    def quantize(self):
        self.use_quantization = True
        self.high_par.weight.data, self.low_par1.weight.data, self.low_par2.weight.data = self.quantized_wts
        self.high_bias, self.low1_bias, self.low2_bias = self.quantized_biases

    def revert(self):
        self.use_quantization = False
        self.high_par.weight.data, self.low_par1.weight.data, self.low_par2.weight.data = self.origin_wts
        self.high_bias, self.low1_bias, self.low2_bias = self.origin_biases

    def forward(self, x, mask, inv_mask, input_fbit):
        extra_bit = 0
        if self.scheme == "uniform":
            extra_bit = 1
        # elif: #extend other schemes
        #     pass

        high_bit_shift = input_fbit + self.wts_fbit + extra_bit - self.biases_fbit
        high = (torch.floor(self.high_par(x) / 2 ** high_bit_shift) + self.high_bias) * mask

        low1_bit_shift = input_fbit + self.wts_fbit + extra_bit - self.biases_fbit
        low = (torch.floor(self.low_par1(x) / 2 ** low1_bit_shift) + self.low1_bias) * inv_mask
        
        low2_bit_shift = self.biases_fbit + self.wts_fbit + extra_bit - self.biases_fbit #because biases_fbit is used in low2 as well;
        low = torch.floor(self.low_par2(low) / 2 ** low2_bit_shift) + self.low2_bias
        
        y = relu_quantize(low + high, 2 ** -self.act_fbit, self.act_nbit, self.biases_fbit)

        return y


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
        # layers for loof
        self.all_layers = nn.Sequential(
            self.first_part, self.reduction, self.mid_part1, self.mid_part2, self.mid_part3, self.mid_part4, self.expansion, self.last_part
        )
        # variables
        ### for storage
        self.origin_wts = []
        self.quantized_wts = []
        self.origin_biases = []
        self.quantized_biases = []
        self.input_nbit, self.input_fbit = 8, 0

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

    def upsample_mask(self, mask):
        mask = mask.repeat(1, self.scale**2, 1, 1)
        mask = F.pixel_shuffle(mask, self.scale)
        inv_mask = torch.where(mask==1, 0, 1)
        return mask, inv_mask

    # 각 레이어의 bias 초기화 (last_part 나중에 할것)
    def init_bias(self, biases):
        bias_list = []
        for key, value in biases.items():
            bias_list.append(value)

        # 매개변수로 받은 bias들을 model 내 변수로 옮기기
        for i in range(len(self.all_layers) - 1):
            self.all_layers[i].high_bias = bias_list[i * 3]
            self.all_layers[i].low1_bias = bias_list[i * 3 + 1]
            self.all_layers[i].low2_bias = bias_list[i * 3 + 2]

            high_bias_size= self.all_layers[i].high_bias.size(dim=0)
            low1_bias_size = self.all_layers[i].low1_bias.size(dim=0)
            low2_bias_size = self.all_layers[i].low2_bias.size(dim=0)

            self.all_layers[i].high_bias = self.all_layers[i].high_bias.view(1, high_bias_size, 1, 1)
            self.all_layers[i].low1_bias = self.all_layers[i].low1_bias.view(1, low1_bias_size, 1, 1)
            self.all_layers[i].low2_bias = self.all_layers[i].low2_bias.view(1, low2_bias_size, 1, 1)

        self.all_layers[-1].bias = bias_list[-1]
        last_bias_size = self.all_layers[-1].bias.size(dim=0)
        self.all_layers[-1].bias = self.all_layers[-1].bias.view(1, last_bias_size, 1, 1)

    #-----------------------quantization------------------------------
    # Functions for assigning quantized weights to the model's weights
    def assign_quantization_settings(self, quantization_settings):
        for i in range(len(self.all_layers)):
            qsi = quantization_settings[i]
            self.all_layers[i].assign_quantization_settings(qsi)

    #nghiant: mode switching
    def quantize(self):
        for li in self.all_layers:
            li.quantize()
            li.cuda()

    # function to revert the model's weights to the original weights
    def revert(self):
        for li in self.all_layers:
            li.revert()
            li.cuda()

    def forward(self, x, th=0.04, dilker=3, dilation=True, eval=False):
        mask, inv_mask = self.create_mask(x, th * 255, dilker, dilation)
        orix = x

        x = self.first_part(x, mask, inv_mask, self.input_fbit)
        x = self.reduction(x, mask, inv_mask, self.first_part.act_fbit)

        x = self.mid_part1(x, mask, inv_mask, self.reduction.act_fbit)
        x = self.mid_part2(x, mask, inv_mask, self.mid_part1.act_fbit)
        x = self.mid_part3(x, mask, inv_mask, self.mid_part2.act_fbit)
        x = self.mid_part4(x, mask, inv_mask, self.mid_part3.act_fbit)

        x = self.expansion(x, mask, inv_mask, self.mid_part4.act_fbit)

        mask, inv_mask = self.upsample_mask(mask)
        y = self.last_part(x, mask, inv_mask, self.expansion.act_fbit)

        y /= 2 ** self.last_part.biases_fbit

        return y