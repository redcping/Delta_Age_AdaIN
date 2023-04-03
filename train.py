import numpy as np
import csv
import math
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from Networks.DAANet import DAA
from EMA import EMA
from datasets.data_utils import DataSetFactory
import cv2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class DAATrainer(object):
    def __init__(self, config):
        self.config = config
        self.set_environment() 
        self.build_model()
        self.set_train_params()   
        self.load_model(self.config.pretrained_fn) 
        self.build_data_loader()
       
        self.save_model_dir = '%s_%s_%s_%s'%(self.config.save_folder,self.config.backbone,str(self.config.num_classes), self.config.da_type)
        try:
            os.mkdir(self.save_model_dir)
        except:
            pass

        self.summary_writer = SummaryWriter(self.save_model_dir)
        self.syth_losses = AverageMeter('SythnLosses')

        

    def set_environment(self):    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in self.config.device_ids])
        
    
    def set_train_params(self):
        self.init_lr    = self.config.lr
        self.lr         = self.init_lr
        self.epochs     = self.config.epochs
        self.optim      = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        self.ema        = EMA(self.model, 0.96)
        self.batch_size = self.config.batch_size
        

    def build_data_loader(self):        
        factory           = DataSetFactory(self.config)
        self.train_loader = DataLoader(factory.training, batch_size=self.batch_size, shuffle=True,
                                         num_workers=self.config.num_works, drop_last=True)
        self.val_loader   = DataLoader(factory.testing, batch_size=self.batch_size, shuffle=True, 
                                           num_workers=self.config.num_works//2, drop_last=True)
        self.val_iter     = iter(self.val_loader)
        
        if self.config.da_type=='image_template':
            self.template_images = factory.template_images.to(self.device).float()
            self.template_labels = factory.template_labels.to(self.device).float()
            if self.config.use_multiple_gpu:
                self.template_images = self.template_images.repeat(len(self.config.device_ids), 1, 1, 1)
                self.template_labels = self.template_labels.repeat(len(self.config.device_ids))


    def build_model(self):
        net_info = {
                    'da_type': self.config.da_type,
                    'feat_dim':self.config.feat_dim,
                    'backbone':self.config.backbone,
                    'num_classes': self.config.num_classes
                   }      
        self.model = DAA(net_info) 
        if self.config.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def load_model(self, model_fn):
        if model_fn=='':
           return
        t = torch.cuda.is_available()
        state_dict = torch.load(model_fn) if t else torch.load(model_fn, map_location=lambda storage, loc: storage)
        try:
            self.optim.load_state_dict(state_dict['optimizer'])
            self.model.load_state_dict(state_dict['net'])
            return 
        except:
            pass
        state_dict = state_dict['net']
        model_dict = self.model.state_dict()
        
        for k,v in state_dict.items():
        #    model_dict[k] = model_dict[k].to(self.device)
            print(k,v.shape)
        ex_list = self.config.pretrained_ex_params
        def ex_fun(k):
            for ex in ex_list:
                if ex in k:
                    return False
            return True
        predict='module.' if self.config.use_multiple_gpu else ''
        pretrained_dict = {k if 'module' in k else predict+k:v for k, v in state_dict.items() if ex_fun(k)}
        model_dict.update(pretrained_dict)
          
        self.model.load_state_dict(model_dict, strict=True)
        print('The model in path %s has been loaded successfully!'%model_fn)

    def save_model(self, epoch):
        self.ema.apply_shadow()
        state = {
                 'net':self.model.state_dict(),
                 'optimizer':self.optim.state_dict()
                 }
        save_fn = '%s/%s_epoch_%d_ac_%s.pth'%(self.save_model_dir,self.config.backbone,epoch,self.accuracy_info)
        torch.save(state, save_fn)
        self.ema.restore()
        print('The model of the %d epoch is successfully stored in path %s!'%(epoch, save_fn))

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 2 every 10 epochs after 20 epoches"""
        self.lr = max(self.init_lr * (1. + np.cos(self.step * np.pi / self.max_iter_step)),2e-6)*0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr 

    def get_val_batch(self):
        try:
            images, labels = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_loader)
            images, labels = next(self.val_iter)
            pass
        return [images, labels]  

    def write_summary(self):
        k=1
        self.summary_writer.add_scalar('train/lr', self.lr, self.step)
        if True:
           
            for key, value in self.summary_image.items():
                self.summary_writer.add_images('{}'.format(key), value[:k], self.step)

            for key, value in self.summary_loss.items():
                self.summary_writer.add_scalar('{}'.format(key), value, self.step)

            for key, value in self.summary_histogram.items():
                self.summary_writer.add_histogram('{}'.format(key), value[:k], self.step)
        #except:
        #    pass

    def train(self):
        pre_epoch = self.config.pre_epoch
        self.max_iter_step = len(self.train_loader) * self.epochs
        self.step = self.config.pre_iter#19890#pre_epoch*len(self.train_loader)/2
       
        print('train begin,total step is %d, total epochs is %d'%(self.max_iter_step-self.step,self.epochs-pre_epoch))
        for epoch in range(pre_epoch,self.epochs+1):
            self.train_epoch(epoch)
            if epoch % 5 == 0:
                self.save_model(epoch)

    def run(self, images, labels, mode='train'):
        images = images.to(self.device)
        for key, value in labels.items():
            labels[key] = value.to(self.device)
        
        run_info = {}
       
        run_info['labels'] = labels['gt_age']
        run_info['mode'] = mode
        run_info['accuracy_threshold'] = self.config.accuracy_threshold
        if self.config.da_type=='image_template':
            run_info['template_images'] = self.template_images
            run_info['template_labels'] = self.template_labels
            
        outputs = self.model(images, run_info)
        if mode.lower()!='test':
            for k, v in outputs['loss'].items(): 
                self.summary_loss['{}/{}'.format(mode, k)]  = v
            self.summary_image['{}/image'.format(mode)] = images[0:1]
        return outputs
        

    def train_epoch(self, epoch):
        self.model.train()
        self.summary_loss={}
        self.summary_image={}
        print('current epoch is %d, learning_rate: %s' %(epoch,str(self.lr)))
        for n, (images, labels) in enumerate(self.train_loader):
            self.step = self.step + 1
            self.adjust_learning_rate(self.optim)
     
            self.summary_loss={}
            self.summary_image={}
            self.summary_histogram={}

            train_outputs = self.run(images,labels,mode='train')
            
            self.total_loss = train_outputs['loss']['total_loss']

            self.optim.zero_grad()
            self.total_loss.backward()
            self.optim.step()
            self.ema.update_params()
            
            self.syth_losses.update(self.total_loss.detach().item(), images.shape[0])
            self.summary_loss['train/avg_loss'] = self.syth_losses.avg

            if n % 50 == 0:
                with torch.no_grad():
                    x_val, y_val = self.get_val_batch()      
                    self.model.eval()                          
                    self.ema.apply_shadow()
                    val_outputs = self.run(x_val, y_val, 'val')
                    self.ema.restore()
                    self.model.train()
                    self.write_summary()                
                    train_accuracy_age = train_outputs['loss']['accuracy'].item()
                    val_accuracy_age   = val_outputs['loss']['accuracy'].item()
                    self.accuracy_info='%.2f-%.2f'%(train_accuracy_age, val_accuracy_age)
                    print('epoch:{},iter:{},total_loss:{},train/val: {}'.format(epoch,self.step,self.total_loss.detach().cpu(), self.accuracy_info))
            del train_outputs,self.total_loss
            
    def test(self):
        self.model.eval()
        cnt, sum_diff = 0, 0
        acc=[0,0,0,0]
        acc_th=[1,3,5,7]
        print('total samples:', len(self.val_loader))
        for n, (x_val, y_val) in enumerate(self.val_loader):
            output = self.run(x_val, y_val, 'test')
            diff = output['l1'].detach().cpu().item()
            #print(diff<=3)
        
            for c in range(len(acc)):
                acc[c] = acc[c] + (1. if diff<=acc_th[c] else 0.)
                
            sum_diff+=diff
            cnt =  cnt + 1
           
            if cnt %1000==0:
                print(cnt, sum_diff/cnt, acc)
               
        print('l1:', sum_diff/cnt)
        print(['ca{}:{}'.format(acc_th[i], acc[i]/cnt) for i in range(len(acc))])


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    from config import Config
    cfg = Config()
    trainer = DAATrainer(cfg)
    if cfg.mode=='test':
        trainer.test() 
    else:
        trainer.train()