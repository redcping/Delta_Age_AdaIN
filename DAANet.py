from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import importlib
from C3AENet import C3AE
from ResNets import ResnetEncoder
from DeltaAdaINNet import DeltaAdaIN
from torch.autograd import Variable



def activate_fn(x,inplace=True):
    return F.relu(x,inplace=inplace)

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride,activate=None):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.activate = activate_fn if activate is None else activate
        
    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return self.activate(x)
        #return activate_fn(x)


       

class DAA(nn.Module):
    def __init__(self, net_info):
        super(DAA,self).__init__()
        
        face_encoder_name = net_info['backbone'].lower()
        
        if face_encoder_name =='C3AE':
            self.da_channels = 32
            self.face_encoder = C3AE()
        elif 'resnet' in face_encoder_name:
            self.da_channels = 512
            self.face_encoder = ResnetEncoder(3, face_encoder_name)
        else:
            pass
        self.num_classes = net_info['num_classes']
        self.da_type = net_info['da_type']
        self.da_operation = DeltaAdaIN(self.num_classes, self.da_type)
        self.feat_dim = net_info['feat_dim']
        #age decoder
        self.age_decoder = nn.Sequential(
                conv(self.da_channels, self.feat_dim, 3, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(self.feat_dim, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _smooth_l1_loss(self, input, target, weight=1.,reduce=True):
        t = torch.abs(input - target)
        ret = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        loss = ret * weight
        if reduce:
            return torch.mean(loss)
        else:
            return loss


    def run_loss(self, ages, labels, accuracy_threshold):
        
        loss={}
        loss['smooth_l1_loss'] = self._smooth_l1_loss(ages, labels.float()) 
                            
        
        age_diff = torch.abs(ages - labels.float())
        loss['l1_loss'] = torch.mean(age_diff)
    
        loss['accuracy'] = 100 * (age_diff <= accuracy_threshold).float().sum()/labels.size(0)
        loss['total_loss'] =   loss['smooth_l1_loss']
        return loss


    def forward(self, x, run_info):
        x =  self.face_encoder(x)
        
        template_labels = torch.arange(self.agenet.num_classes).view(1,-1).to(x)
        template_x      =  None
        if self.agenet.daa_type =='image_template':
            template_labels = run_info['template_labels'].view(1,-1)
            template_x = self.face_encoder(run_info['template_images'])      
        
        da_feats = self.da_operation(x, template_x)      
        b1, b2, c, h, w = da_feats.size()
        delta_ages  = self.age_decoder(da_feats.view(b1*b2, c, h, w)).view(b1, b2)
   
        ages   =  torch.mean(template_labels - delta_ages, -1)
        
        results={}
        results['age'] = ages
        
        if run_info['mode'].lower() != 'test':
            loss=self.run_loss(ages.view(-1), run_info['labels'].view(-1), run_info['accuracy_threshold'])
            results['loss'] =  loss
            
        return results
        


if __name__ == '__main__':
    net_info = {'backbone':'resnet18','num_classes':100,'feat_dim':32, 'da_type':'binary'}
    
    net = DAA(net_info)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    # input_size=(1, 3, 128, 128)
    # x = torch.randn(input_size)
    # # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    # from thop import profile
    # run_info ={'image':x}
    # flops, params = profile(net, inputs=(run_info,))
    # # print(flops)
    # # print(params)
    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fM' % (flops/1000000.0))
    # #x = torch.randn((2,3,64,64))
    # outputs = net(run_info)
