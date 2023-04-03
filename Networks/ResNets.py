from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import importlib
from torch.autograd import Variable


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)



class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
 
    def forward(self, x):
        return self.func(x)
    

class ResnetEncoder(nn.Module):
    def __init__(self, num_in_layers=3, encoder='resnet18', pretrained=False):
        super(ResnetEncoder, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50',\
                           'resnet101', 'resnet152'],\
                           "Incorrect encoder type"
        resnet = class_for_name("torchvision.models", encoder)\
                                (pretrained=False)
        if pretrained:
            #resnet-18
            model_dict = torch.load('./{}.pth'.format(encoder))
            print(encoder,len(model_dict.keys()))
            resnet.load_state_dict(model_dict)


        if num_in_layers == 3:  # Number of input channels
            ks=3
            s=2
            pd=1
            print('firstconv is 3x3')
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                              kernel_size=(ks, ks), stride=(s, s),
                              padding=(pd, pd), bias=False)
            self.firstmaxpool = Lambda(lambda x:x)
        else:
            self.firstconv = resnet.conv1 # H/2
            self.firstmaxpool = resnet.maxpool # H/4

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        # encoder:
        self.encoder1 = resnet.layer1 # H/4
        self.encoder2 = resnet.layer2 # H/8
        self.encoder3 = resnet.layer3 # H/16
        self.encoder4 = resnet.layer4 # H/32

      

    def forward(self, x):
        # encoder
        x_first_conv = self.firstconv(x)
        x = self.firstbn(x_first_conv)
        x = self.firstrelu(x)
        x_pool1 = self.firstmaxpool(x)
        x1 = self.encoder1(x_pool1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return x4
