import torchvision
from utils import *
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import DataLoader

# Dataset:
#  - v, k, and 2 are the same
#  - w and 3 are very similar

t = v2.Compose([
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

# These might be too much
# v2.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0),
# v2.GaussianNoise(mean=0, sigma=0.05), # Noise
# v2.GaussianBlur(kernel_size: Union[int, Sequence[int]], sigma: Union[int, float, Sequence[float]] = (0.1, 2.0)),
# Crop
# JPEG compression
# Erasing
#
# Auto-Augmentation
# v2.AutoAugment(),
# v2.RandAugment(),
# v2.TrivialAugmentWide(),
# v2.AugMix(),

# ds = CombinedImageDataset('./output/combined/labels.csv', './output/combined/images')
# train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)
# train_features, train_labels = next(iter(train_dataloader))
# img = train_features[0].squeeze().transpose(0, 2)
# label = train_labels[0]
# plt.imshow(img)
# plt.show()
# print(f"Label: {label}")

generate_all(t)
combine_generated_all()
