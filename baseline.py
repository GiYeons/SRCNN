from torch import nn
import torch.nn.functional as F
import math as math
from matlab import imresize

class Net(nn.Module):
    def __init__(self, scale, in_channel=1, num_filters1=16, num_filters2=32):
        super(Net, self).__init__()
        self.scale = scale
        self.fk, self.rk,  self.mk, self.tk = 5, 1, 3, 9     # kernel size

        self.first_part = nn.Conv2d(in_channels=in_channel, out_channels=num_filters2, kernel_size=self.fk, padding=self.fk//2)
        self.reduction = nn.Conv2d(in_channels=num_filters2, out_channels=num_filters1, kernel_size=self.rk, padding=self.rk//2)

        self.mid_part1 = nn.Conv2d(in_channels=num_filters1, out_channels=num_filters1, kernel_size=self.mk, padding=self.mk//2)
        self.mid_part2 = nn.Conv2d(in_channels=num_filters1, out_channels=num_filters1, kernel_size=self.mk, padding=self.mk//2)
        self.mid_part3 = nn.Conv2d(in_channels=num_filters1, out_channels=num_filters1, kernel_size=self.mk, padding=self.mk//2)
        self.mid_part4 = nn.Conv2d(in_channels=num_filters1, out_channels=num_filters1, kernel_size=self.mk, padding=self.mk//2)

        self.expansion = nn.Conv2d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=self.rk, padding=self.rk//2)
        self.last_part = nn.ConvTranspose2d(
            in_channels=num_filters2, out_channels=in_channel, kernel_size=self.tk, padding=self.tk//2, stride=scale, output_padding=scale-1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.first_part(x))
        x = self.relu(self.reduction(x))

        x = self.relu(self.mid_part1(x))
        x = self.relu(self.mid_part2(x))
        x = self.relu(self.mid_part3(x))
        x = self.relu(self.mid_part4(x))

        x = self.relu(self.expansion(x))
        y = self.relu(self.last_part(x))

        return y

