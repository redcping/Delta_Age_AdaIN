from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class BRA(nn.Module):
    def __init__(self, in_channels):
        super(BRA, self).__init__()
        self.bra = nn.Sequential(
            nn.BatchNorm2d(in_channels), Hswish(), nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        return self.bra(x)


class se_module(nn.Module):
    def __init__(self, in_channels):
        super(se_module, self).__init__()
        self.se_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            Hswish(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        sig = self.se_conv(x)
        return x * sig


class C3AE(nn.Module):
    def __init__(self, in_channels=3):
        super(C3AE, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3),
            BRA(32),  # H/2
            se_module(32),
            nn.Conv2d(32, 32, 3),
            BRA(32),  # H/4
            se_module(32),
            nn.Conv2d(32, 32, 3),
            BRA(32),  # H/8
            se_module(32),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            Hswish(),
            se_module(32),
            nn.Conv2d(32, 32, 1),
            se_module(32),
            # nn.Conv2d(32, 128, 3),
            # nn.Dropout(self.dropout)
        )

        self._initialize_weights()

    def forward(self, x):
        return self.feature(x)

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# if __name__ == '__main__':
#     net = Network()
#     print('Network:\n', net)
#     print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
#     input_size=(1, 3, 64, 64)
#     x = torch.randn(input_size)
#     # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
#     from thop import profile
#     flops, params = profile(net, inputs=(x,))
#     # print(flops)
#     # print(params)
#     print('Total params: %.2fM' % (params/1000000.0))
#     print('Total flops: %.2fM' % (flops/1000000.0))
#     x = torch.randn((2,3,64,64))
#     outputs = net(x)
