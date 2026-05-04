# to
# Map(character, image_array)

import os
import string
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2, InterpolationMode
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                self.images.append((os.path.basename(root), f))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.images[idx][0]
        fname = self.images[idx][1]
        image = decode_image(os.path.join(self.img_dir, label, fname))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def rename_folder():
    dir = './asl-alphabet-test'
    for subdir in os.listdir(dir):
        letter = subdir.lower()
        os.rename(os.path.join(dir, subdir), os.path.join(dir, letter))

    dir = './hand-sign-gesture-dataset-az-09-25k-images'
    for subdir in os.listdir(dir):
        letter = subdir.split('-')[0].lower()
        os.rename(os.path.join(dir, subdir), os.path.join(dir, letter))

def gen():
    dataset = CustomImageDataset('./asl-alphabet-test')
    im1 = dataset.__getitem__(0)
    t = v2.Compose([
        v2.ToImage(),
        v2.RandomAffine((-30, 30), translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0, center=None), # Rotation, Translation, Scale
        v2.GaussianNoise(mean=0, sigma=0.05), # Noise
        # v2.ColorJitter(
        #     brightness: Optional[Union[float, Sequence[float]]] = None,
        #     contrast: Optional[Union[float, Sequence[float]]] = None,
        #     saturation: Optional[Union[float, Sequence[float]]] = None,
        #     hue: Optional[Union[float, Sequence[float]]] = None
        # ) # Brightness, Contrast, Saturation, Hue
        # v2.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0)
        # v2.RandomHorizontalFlip(p=0.5)
        # v2.RandomVerticalFlip(p=0.5)
        # v2.GaussianBlur(kernel_size: Union[int, Sequence[int]], sigma: Union[int, float, Sequence[float]] = (0.1, 2.0))

        # Crop
        # JPEG compression lmao (our data is already JPG tho)
        # Erasing

        # "Auto-Augmentation"
        # v2.AutoAugment()
        # v2.RandAugment()
        # v2.TrivialAugmentWide()
        # v2.AugMix()
    ])

    augmented_im = t(im1[0]).transpose(0, 2)
    plt.imshow(augmented_im)
    plt.axis('off')
    plt.show()

gen()
