import os
shape=(128,128)
import torchvision.transforms as transforms
class Config:
    def __init__(self):
        # data
        self.datanames = "utkface"
        self.data_folder = "./datasets/"
        self.do_aug = True
        self.num_works = 4

        #utils
        self.model_purpose=""
        self.validation_transformations = [
                            transforms.Resize(shape),
                transforms.ToTensor(),

        ]
        self.training_transformations = [
            transforms.Resize(shape),
            transforms.ToTensor(),

        ]
        # train
        self.batch_size = 64
        self.lr = 1e-3
        self.epochs = 120
        self.use_multiple_gpu = False
        self.device_ids = [0, 1] if self.use_multiple_gpu else [0]
        self.pre_epoch = 0
        self.pre_iter = 0
        self.accuracy_threshold = 7
        self.do_multiscale = False
        self.mode = "train" if self.batch_size > 1 else "test"
        # self.save_folder = "/content/gdrive/MyDrive/CV_DAA_PROJ/DAA_models/{}_{}".format(self.model_purpose,self.datanames)

        # net
        self.backbone = "resnet18"  # ['c3ae','resnet18']
        self.input_size = 128  # 96

        self.feat_dim = 32
        self.min_age = 1
        self.max_age = 100
        self.num_classes = self.max_age - self.min_age + 1
        self.da_type = "binary"  # ['binary','decimal', 'image_template']
        self.image_template_path = ""

        self.pretrained_fn = ""
        self.pretrained_ex_params = []

    def save_model_folder(self):
        self.save_folder = "/content/gdrive/MyDrive/CV_DAA_PROJ/DAA_models/{}_{}".format(self.model_purpose,self.datanames)