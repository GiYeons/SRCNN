from torch import nn
import torch.nn.functional as F
import math as m
from matlab import imresize

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, kernel_size=9, out_channels=64, padding=9//2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, kernel_size=5, out_channels=32, padding=5//2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, kernel_size=5, out_channels=1, padding=5//2)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, mean=0.0, std=0.001)
        #         nn.init.zeros_(m.bias)


    def forward(self, x, scale, border_cut=None):
        x = F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False)  # lr image interpolation(preprocess)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if(border_cut!=None):
            x = x[:, :, border_cut:-border_cut, border_cut:-border_cut]

        return x

