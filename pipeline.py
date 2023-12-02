import torchvision.transforms as transforms
import numpy as np
import csv
import math
import random
from tensorboardX import SummaryWriter
import warnings
import logging
warnings.filterwarnings("ignore")
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
import matplotlib.pyplot as plt


shape=(128,128)

class Pipeline():
    def __init__(
        self,
        seed,
        shape,
        test_configs
    ):
        self.set_seed(seed)
        self.set_datetime()
        self.test_configs = test_configs

    def plot_train_val_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.trainer.epoch_train_losses, label='Training Loss')
        plt.plot(self.trainer.epoch_val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.trainer.save_model_dir}/train_val_loss_{self.current_datetime}.png')


    def plot_train_val_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.trainer.epoch_train_accuracies, label='Training Accuracy')
        plt.plot(self.trainer.epoch_val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracies Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{self.trainer.save_model_dir}/train_val_accuracy_{self.current_datetime}.png')

    def plot_test_l1_losses(self,l1_losses):
        interval = 10
        avg_l1_losses = [sum(l1_losses[i:i+interval])/interval for i in range(0, len(l1_losses), interval)]
        plt.figure(figsize=(10, 5))
        plt.plot(avg_l1_losses)
        plt.title('Average L1 Loss per Interval')
        plt.xlabel('Interval')
        plt.ylabel('Average L1 Loss')
        plt.savefig(f'{self.trainer.save_model_dir}/test_l1_loss_{self.current_datetime}.png')


    def training_pipeline(self):
        logging.info("\n\n\n BEGIN TRAINING \n\n")
        self.trainer.train()

        self.plot_train_val_loss()
        self.plot_train_val_accuracy()

    def testing_pipeline(self):
        logging.info("\n\n\n BEGIN TESTING \n\n")
        l1_losses = self.trainer.test()
        self.plot_test_l1_losses(l1_losses)

    def set_seed(self,seed):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_datetime(self):
        current_datetime = datetime.now()
        self.current_datetime = current_datetime.strftime("%m-%d_%H")

    def run(self):
        for config_instance in self.test_configs:

            self.cfg = Config()
            for key, value in config_instance.items():
                setattr(self.cfg, key, value)

            self.cfg.save_model_folder()
            self.trainer = DAATrainer(self.cfg)

            print(f"save_model_dir: {self.trainer.save_model_dir}")

            os.makedirs(self.trainer.save_model_dir, exist_ok=True)
            log_file_path = os.path.join(self.trainer.save_model_dir, "session.log")
            logging.basicConfig(
                filename=log_file_path,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                force=True  # Force applying this configuration
            )

            print(self.cfg.__dict__)
            print(self.cfg.mode)

            if self.cfg.mode == "test":
                self.testing_pipeline()
            else:
                self.training_pipeline()
                self.testing_pipeline()

def custom_gaussian_blur(image):
    # Define your custom Gaussian blur operation here
    # Example: Applying Gaussian blur using OpenCV
    return cv2.GaussianBlur(np.array(image), (3, 3), 2)

test_configs_megaage_asian50 = [
    {
        "model_purpose":"no_augmentation_VF_50epoch",
        "epochs":50,
        "datanames":"megaage_asian",

        "do_aug": False,
        "batch_size": 64,
        "lr":1e-3,
        "input_size": 128,
        "backbone": "resnet18",
        "feat_dim": 32,
        "validation_transformations":[
            transforms.Resize(shape),
            transforms.ToTensor()
        ],
        "training_transformations":[
            transforms.Resize(shape),
            transforms.ToTensor()
        ],
    },
    {
        "model_purpose":"color_based_aug_VF_50epoch",
        "epochs":50,
        "datanames":"megaage_asian",

        "do_aug": True,
        "batch_size": 64,
        "lr":1e-3,
        "input_size": 128,
        "backbone": "resnet18",
        "feat_dim": 32,
        "validation_transformations":[
            transforms.Resize(shape),
            transforms.ToTensor()
        ],
        "training_transformations":[
            transforms.Resize(shape),

            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)],  p=0.20),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)],  p=0.20),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.2)],  p=0.20),
            transforms.RandomGrayscale(0.05),

            transforms.ToTensor()

        ],

    },
    {
        "model_purpose":"distortion_based_aug_VF_50epoch",
        "epochs":50,
        "datanames":"megaage_asian",

        "do_aug": True,
        "batch_size": 64,
        "lr":1e-3,
        "input_size": 128,
        "backbone": "resnet18",
        "feat_dim": 32,

        "validation_transformations":[
            transforms.Resize(shape),
            transforms.ToTensor(),
        ],
        "training_transformations":[
            transforms.Resize(shape),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomApply([custom_gaussian_blur], p=0.2),
            transforms.ToTensor(),
            transforms.Resize((128, 128))  # Ensure consistent size for all images


        ]
    },
    {
        "model_purpose":"distortion_color_combined_aug_VF_50epoch",
        "epochs":50,
        "datanames":"megaage_asian",

        "do_aug": True,
        "batch_size": 64,
        "lr":1e-3,
        "input_size": 128,
        "backbone": "resnet18",
        "feat_dim": 32,

        "validation_transformations":[
            transforms.Resize(shape),
            transforms.ToTensor(),
        ],
        "training_transformations":[
            transforms.Resize(shape),

            # Color based
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)],  p=0.20),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)],  p=0.20),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.2)],  p=0.20),
            transforms.RandomGrayscale(0.05),

            # distortion based
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomApply([custom_gaussian_blur], p=0.2),

            transforms.ToTensor(),
            transforms.Resize((128, 128))

        ]
    }
]

if __name__ == "__main__":
	utkface_pipeline = Pipeline(
		seed=42,
		shape=(128,128),
		test_configs= test_configs_megaage_asian50[2:]
	)
	utkface_pipeline.run()