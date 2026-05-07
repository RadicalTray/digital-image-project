import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
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

ds = ImageDataset('./output/combined/labels.csv', './output/combined/images')
dataloader = DataLoader(ds, batch_size=64, shuffle=True)
features, labels = next(iter(dataloader))

fig, ax = plt.subplots(nrows=8, ncols=8, squeeze=False)
for i in range(8):
    for j in range(8):
        img = features[i * 8 + j].squeeze().transpose(0, 2)
        label = labels[i * 8 + j]
        ax[i, j].imshow(img)
        ax[i, j].axis('off')
        ax[i, j].set_title(label)
plt.show()
