import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import csv
import random
import math
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
import cv2
import os, io
from torch.utils.data import DataLoader


class DataSetFactory:
    def __init__(self, config):
        self.config = config
        if self.config.data_folder[-1] != "/":
            self.config.data_folder = self.config.data_folder + "/"

        samples = {}
        samples = self.read_age()

        print(
            "training size %d : testing size %d"
            % (len(samples["training"]), len(samples["testing"]))
        )
        shape = (self.config.input_size, self.config.input_size)

        val_transform = transforms.Compose(
            [
                transforms.Resize(shape),
                transforms.ToTensor(),
            ]
        )
        train_transform = transforms.Compose(
            [
                transforms.Resize(shape),
                transforms.RandomGrayscale(0.1),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                # transforms.RandomRotation(degrees=(20)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        train_transform = train_transform if self.config.do_aug else val_transform

        if self.config.da_type == "image_template":
            if os.path.exists(self.config.image_template_path):
                info = np.load(self.config.image_template_path)
                images = info["rgb"]
                labels = info["labels"]
                self.template_images = torch.tensor(images)
                self.template_labels = torch.tensor(labels)
            else:
                select_idxs = samples["select_idxs"]
                # print(select_idxs)
                labels, images = [], []
                for select in select_idxs:
                    label, idx = select
                    image_fn = samples["training"][idx]["image"]
                    rgb = Image.open(image_fn).convert("RGB")
                    rgb = val_transform(rgb)[None, ...]
                    labels.append(label)
                    images.append(rgb)
                labels = np.array(labels)
                images = torch.cat(images, 0).numpy()
                image_template_path = "{}{}".format(
                    self.config.data_folder, self.config.datanames.replace(",", "_")
                )
                np.savez_compressed(image_template_path, rgb=images, labels=labels)
                print("image_template path is: ", image_template_path)
                self.template_images = torch.tensor(images)
                self.template_labels = torch.tensor(labels)

        self.training = DataSet(
            transform=train_transform,
            samples=samples["training"],
            type_="training",
            resize_shape=self.config.input_size,
        )
        self.testing = DataSet(
            transform=val_transform,
            samples=samples["testing"],
            type_="testing",
            resize_shape=self.config.input_size,
        )

        print("dataset---->ok!")

    def random_choose_template(self, smaples_idx):
        select_idxs = []
        for i in range(self.config.num_classes):
            idxs = smaples_idx[i]
            if len(idxs) == 0:
                continue
            # np.random.choice(range(len(idxs)), size = 1, replace=False)
            select_idxs.append([i, np.random.randint(len(idxs))])
        return select_idxs

    def read_age(self):
        samples = {}
        samples["training"] = []
        samples["testing"] = []
        age_samples = [[] for k in range(self.config.num_classes)]
        for name in self.config.datanames.split(","):
            filename = self.config.data_folder + name + "/" + name + ".csv"

            with open(filename, "r") as csvin:
                data = csv.reader(csvin)
                next(data)
                for row in data:
                    age = int(float(row[0]) + 0.5)
                    if (
                        age < 0
                        or age > 100
                        or name == "imdb_wiki"
                        and (age < 6 or age > 90)
                    ):
                        continue
                    age = max(age, self.config.min_age)
                    age = min(age, self.config.max_age)
                    age = age - self.config.min_age
                    sample = {"gt_age": age}

                    # row[2]: data path
                    image_fn = row[1]
                    sample["image"] = (
                        image_fn
                        if self.config.data_folder in image_fn
                        else self.config.data_folder + image_fn
                    )
                    kk = row[2]  # ['training' or 'testing']
                    samples[kk].append(sample)
                    if kk == "training":
                        age_samples[age].append(len(samples[kk]) - 1)

        if self.config.da_type == "image_template" and not os.path.exists(
            self.config.image_template_path
        ):
            samples["select_idxs"] = self.random_choose_template(age_samples)

        for k in range(self.config.num_classes):
            print(k, len(age_samples[k]))

        return samples


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self, transform=None, samples=None, type_="training", resize_shape=None
    ):
        self.transform = transform
        self.samples = samples
        self.resize_shape = resize_shape
        self.type_ = type_

    def crop_and_resize_data(self, img, s=8):
        width, height = img.size
        w = max(width, s)
        d = np.random.randint(1, 2 * w // s) if self.type_ == "training" else w // s

        new_min_x, new_min_y = max(d, 0), max(d, 0)
        new_max_x, new_max_y = min(w - d, width), min(w - d, height)
        box = (new_min_x, new_min_y, new_max_x, new_max_y)
        ratio_w = 1.0 / np.float32(new_max_x - new_min_x)
        ratio_h = 1.0 / np.float32(new_max_y - new_min_y)
        box = (new_min_x, new_min_y, new_max_x, new_max_y)
        out_img = img.crop(box)
        return out_img

    def __getitem__(self, index):
        sample = self.samples[index]
        image_fn = sample["image"]
        labels = {}
        rgb = Image.open(image_fn).convert("RGB")

        rgb = self.crop_and_resize_data(rgb)
        rgbs = self.transform(rgb)

        # print(sample.keys())
        for k, v in sample.items():
            if k not in ["gt_box", "image"]:
                labels[k] = torch.tensor(v).float()
        return rgbs, labels

    def __len__(self):
        return len(self.samples)
