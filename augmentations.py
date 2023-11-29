import torch
import numpy as np


class ElasticTransform:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, x):
        x = np.array(x)
        shape = x.shape
        dx = torch.tensor(np.random.randn(*shape) * self.alpha)
        dy = torch.tensor(np.random.randn(*shape) * self.alpha)
        blurred = F.gaussian_blur(x, (0, 0), self.sigma)
        return torch.tensor(x + dx, dtype=torch.float32), torch.tensor(
            y + dy, dtype=torch.float32
        )


transforms.Compose(
    [
        ElasticTransform(
            alpha=10.0, sigma=3.0
        )  # Apply elastic transformations with specified parameters
    ]
)


##### Color Jitter #####
transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ]
)

##### noise injection #####
import torchvision.transforms.functional as F


def add_noise(x, mean=0, std=1):
    noise = torch.randn_like(x) * std + mean
    return x + noise


transforms.Compose(
    [
        transforms.Lambda(
            lambda x: add_noise(x, mean=0, std=0.1)
        )  # Add Gaussian noise with mean 0 and std 0.1
    ]
)

##### GaussianBlur #####
from torchvision.transforms import GaussianBlur

transforms.Compose(
    [GaussianBlur(kernel_size=3)]  # Apply Gaussian blur with a kernel size of 3
)

##### brightness and contrast #####
transforms.Compose(
    [
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2
        ),  # Randomly adjust brightness and contrast
    ]
)

##### cropping and zooming #####
transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=224, scale=(0.8, 1.0)
        ),  # Randomly crop and resize the image
    ]
)

##### Rotation and Flipping #####
transforms.Compose(
    [
        transforms.RandomRotation(
            degrees=30
        ),  # Rotate the image randomly up to 30 degrees
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Randomly flip the image horizontally with a 50% probability
    ]
)
