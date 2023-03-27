import os

class Config:
    def __init__(self):
        # train
        self.batch_size = 128
        self.lr = 1e-3
        self.epochs = 200
        self.use_multiple_gpu=False
        self.device_ids=[0,1] if self.use_multiple_gpu else [0]
        self.pre_epoch = 0
        self.pre_iter = 0
        self.accuracy_threshold = 1
        self.do_multiscale=False
        self.mode='train'
        self.save_folder='/media/redcping/data/ages/models/'
        
        #net
        self.backbone = 'resnet18' #['c3ae','resnet18']
        self.input_size = 128 #96
        
        self.feat_dim=32
        self.min_age = 1 #幼儿
        self.max_age = 100#老年
        self.num_classes = self.max_age - self.min_age + 1
        self.da_type = 'binary' #['binary','decimal', 'image_template']
        self.save_flag = self.backbone

        self.pretrained_fn=''
        self.pretrained_ex_params=[]
        #data
        self.datanames = 'megaage_asian'
        self.data_folder = '/media/redcping/data/ages/'
        self.do_aug = True
        self.num_works = 4
