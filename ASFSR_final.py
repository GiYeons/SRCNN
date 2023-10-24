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

        self.sub_pixel = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=self.out_c, kernel_size=self.ker, padding=self.ker // 2, bias=False),
                                nn.PixelShuffle(scale))

        # custom bias
        self.bias = None

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
                # nn.init.zeros_(m.bias.data)

        # ===========
        self.ones = nn.Parameter(data=torch.ones(size=(in_c, 1, 1, 1)).float(), requires_grad=False)
        # ===========

    def forward(self, x, mask, inv_mask, bit_shift):
        # original code
        """

        high = self.high_par(x) * mask
        low = self.low_par1(x)
        low = self.low_par2(low) * inv_mask
        """
        x = (self.sub_pixel[0](x) / 2**bit_shift) + self.bias
        result = self.sub_pixel[1](x)

        return result


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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                # nn.init.zeros_(m.bias.data)

    def forward(self, x, mask, inv_mask, bit_shift):
        high = (self.high_par(x) / 2**bit_shift + self.high_bias) * mask
        low = (self.low_par1(x) / 2**bit_shift + self.low1_bias) * inv_mask
        low = self.low_par2(low) / 2**bit_shift + self.low2_bias

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
        # layers for loof
        self.all_layers = nn.Sequential(
            self.first_part, self.reduction, self.mid_part1, self.mid_part2, self.mid_part3, self.mid_part4, self.expansion, self.last_part
        )
        # variables
        self.wts_nbit, self.wts_fbit = 8, 4
        self.biases_nbit, self.biases_ibit = 8, 4
        self.biases_fbit = self.biases_nbit - self.biases_ibit
        self.bit_shift, self.output_fbit = 0, 0
        self.act_nbit, self.act_fbit = 8, 4
        self.scales_nbit, self.scales_ibit = 1, 1
        self.origin_wts = []
        self.quantized_wts = []
        self.origin_biases = []
        self.quantized_biases = []



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
    def quantize(self, scheme="uniform", wts_nbit=8, wts_fbit=4):
        self.prepare_q_weight(scheme, wts_nbit, wts_fbit)

        for i in range(len(self.all_layers) - 1):
            self.all_layers[i].high_par.weight.data = self.quantized_wts[i][0]
            self.all_layers[i].low_par1.weight.data = self.quantized_wts[i][1]
            self.all_layers[i].low_par2.weight.data = self.quantized_wts[i][2]

            # if(len(self.quantized_biases) > 0):
            #     self.all_layers[i].high_bias = self.quantized_biases[i][0]
            #     self.all_layers[i].low1_bias = self.quantized_biases[i][1]
            #     self.all_layers[i].low2_bias = self.quantized_biases[i][2]
            self.all_layers[i].cuda()

        # last layer
        self.all_layers[-1].sub_pixel[0].weight.data = self.quantized_wts[-1][0]

        # if(len(self.quantized_biases) > 0):
        #     self.all_layers[-1].bias = self.quantized_biases[-1][0]

        self.all_layers[-1].cuda()


    # function to revert the model's weights to the original weights
    def revert(self):
        for i in range(len(self.all_layers) - 1):
            self.all_layers[i].high_par.weight.data = self.origin_wts[i][0]
            self.all_layers[i].low_par1.weight.data = self.origin_wts[i][1]
            self.all_layers[i].low_par2.weight.data = self.origin_wts[i][2]
            self.all_layers[i].cuda()

    # function for generating quantized weights
    def prepare_q_weight(self, scheme="uniform", wts_nbit=8, wts_fbit=4):
        self.wts_nbit = wts_nbit
        self.wts_fbit = wts_fbit

        #TConv에서 high, low weight를 없앴으므로 따로 처리해야 함 (작성예정)
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

            # biases
            high_b = self.all_layers[i].high_bias
            low_b1 = self.all_layers[i].low1_bias
            low_b2 = self.all_layers[i].low2_bias

            if(self.biases_nbit > 0):
                high_biases = self.quantize_and_constrain(high_b, self.biases_nbit, self.biases_ibit)
                low_biases1 = self.quantize_and_constrain(low_b1, self.biases_nbit, self.biases_ibit)
                low_biases2 = self.quantize_and_constrain(low_b2, self.biases_nbit, self.biases_ibit)

                self.origin_biases.append([high_b, low_b1, low_b2])
                self.quantized_biases.append([high_biases, low_biases1, low_biases2])


        # last_part
        last_wts = self.all_layers[-1].sub_pixel[0].weight.data
        wts_step = 2 ** -wts_fbit

        print('step: ', wts_step)

        if(scheme == "none"):
                pass

        elif (scheme == "uniform"):
            last_weight, _ = self.uniform_quantize(last_wts, wts_step, wts_nbit)

        self.origin_wts.append([last_wts, None, None])
        self.quantized_wts.append([last_weight, None, None])

        ## biases
        last_b = self.all_layers[-1].bias

        if (self.biases_nbit > 0):
            last_biases = self.quantize_and_constrain(last_b, self.biases_nbit, self.biases_ibit)

            self.origin_biases.append([last_b, None, None])
            self.quantized_biases.append([last_biases, None, None])


        # bit shift
        self.bit_shift, self.output_fbit = self.calculate_bit_shift(8, 4)
        print(self.bit_shift, self.output_fbit)

        # print(self.all_layers[0].high_par.weight.data)

    def uniform_quantize(self, x, step, nbit):
        pos_end = 2 ** nbit - 1     # 255
        neg_end = -pos_end          # -255

        output = 2 * torch.round(x / step + 0.5) - 1
        output = torch.clip(output, min=neg_end, max=pos_end)

        output_store = (output - 1) / 2

        return output, output_store

    def scale_linear_quantize(self, wts, n_out_channel, n_coff):
        nwts = torch.numel(wts)
        nwts_per_channel = nwts / n_out_channel

        scale_factor = torch.zeros(n_out_channel)

        # should be modified

    # bias & scale quantization
    def quantize_and_constrain(self, x, nbit, ibit, sign=True):
        qfactor = 2 ** (nbit-ibit)
        nfactor = 2 ** nbit

        if(sign):
            max = nfactor / 2 - 1
            min = max - nfactor + 1
        else:
            max = nfactor - 1
            min = 0

        output = torch.round(x * qfactor)
        output = torch.clip(output, min=min, max=max)

        return output


    def calculate_bit_shift(self, input_ibit, input_fbit):
        scheme = "uniform"  # test
        activation = "float_relu"

        if(scheme == "none"):
            wts_fbit = 0
        elif(scheme == "uniform"):
            wts_fbit = self.wts_fbit + 1

        scales_fbit = self.scales_nbit - self.scales_ibit
        biases_fbit = self.biases_nbit - self.biases_ibit

        bit_shift = input_fbit + wts_fbit + scales_fbit - biases_fbit

        if(activation == "float_relu"):
            act_fbit = biases_fbit

        output_fbit = act_fbit
        input_fbit = act_fbit

        return bit_shift, output_fbit

    def relu_quantize(self, x, step, nbit, bias_shift):
        pos_end = torch.tensor(2 ** nbit - 1).to(x.device)
        # print("relu 전:", x)

        activations = x;
        activations = torch.where(activations >= 0, activations / round(2 ** bias_shift * (step)), torch.zeros_like(activations))
        activations = torch.where(activations > pos_end, pos_end.float(), activations)

        return activations


    def forward(self, x, th=0.04, dilker=3, dilation=True, eval=False):

        mask, inv_mask = self.create_mask(x, th, dilker, dilation)

        orix = x

        x = self.relu_quantize(self.first_part(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        # insert activation quantization function?
        x = self.relu_quantize(self.reduction(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)

        x = self.relu_quantize(self.mid_part1(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        x = self.relu_quantize(self.mid_part2(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        x = self.relu_quantize(self.mid_part3(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
        x = self.relu_quantize(self.mid_part4(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)

        x = self.relu_quantize(self.expansion(x, mask, inv_mask, self.bit_shift), 2**-self.act_fbit, self.act_nbit, self.biases_fbit)
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
        y = self.last_part(x, mask, inv_mask, self.bit_shift)



        return y

