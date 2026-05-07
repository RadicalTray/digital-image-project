import os
import csv
import json
import stat
import shutil
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import decode_image
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.functional import to_pil_image

class OriginalImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        assert os.path.isdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                self.images.append((os.path.basename(root), f))
        # On my machine, os.walk finds non-existent files (they only exist during os.walk or something)
        # So we must filter them out.
        l = []
        for i, im in enumerate(self.images):
            path = os.path.join(img_dir, im[0], im[1])
            if not os.path.exists(path):
                print('non-existent path:', path)
                l.append(i)
        l = reversed(sorted(l)) # l should already have been sorted but eh
        for i in l: self.images.pop(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.images[idx][0]
        fname = self.images[idx][1]
        path = os.path.join(self.img_dir, label, fname)
        image = decode_image(path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class CombinedImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
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

def plot(original, augmented):
    assert len(original) == len(augmented)
    fig, ax = plt.subplots(nrows=2, ncols=len(original), squeeze=False)
    for i, im in enumerate(original):
        ax[0, i].imshow(im)
    for i, im in enumerate(augmented):
        ax[1, i].imshow(im)
    plt.show()

def show():
    images = [asl[500][0], hsg[500][0]]
    original = [x.transpose(0, 2) for x in images]
    try:
        while True:
            plot(original, [t(x).transpose(0, 2) for x in images])
    except KeyboardInterrupt:
        exit(0)

def generate(transform, dataset, output_dir):
    num_augmented = 4
    for i, (image, label) in enumerate(dataset):
        print(f'[{i+1}/{len(dataset)}]', os.path.join(dataset.img_dir, dataset.images[i][0], dataset.images[i][1]))
        dir = os.path.join(output_dir, label)
        os.makedirs(dir, exist_ok=True)

        # FORGOT RESIZE
        # to_pil_image(image).save(os.path.join(dir, f'{i}.jpg'))

        for j, augmented in enumerate([transform(image) for x in range(num_augmented)]):
            output_path = os.path.join(output_dir, label, f'{i}.jpg')
            to_pil_image(augmented).save(os.path.join(dir, f'{i}-augmented-{j}.jpg'))

def generate_all(transform, output_dir='./output'):
    asl = OriginalImageDataset('./asl-alphabet-test')
    hsg = OriginalImageDataset('./hand-sign-gesture-dataset-az-09-25k-images')
    generate(transform, asl, os.path.join(output_dir, 'asl'))
    generate(transform, hsg, os.path.join(output_dir, 'hsg'))

def combine_generated(dataset, output_dir, prefix):
    data = []
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    for i in range(len(dataset)):
        label, fname = dataset.images[i]
        img_path = os.path.join(f'{prefix}-{fname}')
        src = os.path.join(dataset.img_dir, label, fname)
        dst = os.path.join(img_dir, img_path)
        shutil.copyfile(src, dst)
        data.append({'image': img_path, 'label': label})
    return data

def combine_generated_all(output_dir='./output'):
    asl = OriginalImageDataset(os.path.join(output_dir, 'asl'))
    hsg = OriginalImageDataset(os.path.join(output_dir, 'hsg'))
    output_dir = os.path.join(output_dir, 'combined')
    data = []
    data.extend(combine_generated(asl, output_dir, 'asl'))
    data.extend(combine_generated(hsg, output_dir, 'hsg'))
    with open(os.path.join(output_dir, 'labels.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label'])
        writer.writeheader()
        writer.writerows(data)

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
