# to
# Map(character, image_array)

import os
import string

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

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
    im1 = Image.open('./asl-alphabet-test/a/A0001_test.jpg')
    t = v2.Compose([
        v2.ToImage(),
        v2.RandomAffine((-30, 30)), # Rotation
        v2.GaussianNoise(sigma=0.05), # Noise

        # Rotation
        # Translation
        # Resize
        # Shrink (Zoom out)
        # Brightness
        # Noise
        # Blur
        # Perspective
        # Flips
        # Crop
        # JPEG compression lmao (our data is already JPG tho)

        # Erasing

        # "Auto-Augmentation"
        # v2.AutoAugment()
        # v2.RandAugment()
        # v2.TrivialAugmentWide()
        # v2.AugMix()
    ])

    augmented_im = t(im1).numpy().transpose(1, 2, 0)
    plt.imshow(augmented_im)
    plt.axis('off')
    plt.show()

gen()
