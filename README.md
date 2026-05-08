- Augmentation

```py
from torchvision.transforms import v2, InterpolationMode

transform = v2.Compose([
    v2.ToImage(),
    # Resize. asl-alphabet-test is 200x200 while the other is 224x224
    v2.Resize((224, 224)),
    # Rotation, Translation, Scale
    v2.RandomAffine(
        (-180, 180), # Rotation
        translate=(0.2, 0.2),
        scale=(0.5, 1.5),
        shear=None,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
        center=None,
    ),
    # Flip. Left Hand and Right Hand
    v2.RandomHorizontalFlip(p=0.5),
    # Brightness. The dataset with numbers had only dark skin hand
    v2.ColorJitter(
        brightness=0.5,
        contrast=None,
        saturation=None,
        hue=None,
    ),
])
```

- Loading Dataset
```py
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

ds = ImageDataset('./dataset/labels.csv', './dataset/images')
dataloader = DataLoader(ds, batch_size=64, shuffle=True)

for features, labels in dataloader:
    fig, ax = plt.subplots(nrows=8, ncols=8, squeeze=False)
    for i in range(8):
        for j in range(8):
            img = features[i * 8 + j].squeeze().transpose(0, 2)
            label = labels[i * 8 + j]
            ax[i, j].imshow(img)
            ax[i, j].axis('off')
            ax[i, j].set_title(label)
    plt.show()
```
