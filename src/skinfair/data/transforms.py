"""Image transforms. Owner: Afzal Raza."""
from torchvision import transforms as T

# Per-channel (R, G, B) mean and std of ImageNet. The pretrained models were trained on
# images normalised with these values, so our inputs must be normalised the same way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(size=224):
    """Random augmentation pipeline for training images.

    Input: output side length in pixels. Returns a transform: PIL image -> tensor [3, size, size].
    """
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),  # random crop of 70-100% of area, resized
        T.RandomHorizontalFlip(),  # lesions have no fixed orientation, so flips are safe
        T.RandomVerticalFlip(),
        T.RandomRotation(20),  # rotate by a random angle in [-20, 20] degrees
        # hue=0 on purpose: hue shifts change apparent skin tone, which would confound the
        # per-Fitzpatrick fairness analysis. Brightness/contrast/saturation only.
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        T.ToTensor(),  # PIL [H, W, 3] in 0-255 -> float tensor [3, H, W] in 0-1
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # (x - mean) / std per channel
    ])


def eval_transforms(size=224):
    """Deterministic pipeline for val/test images (no randomness).

    Input: output side length. Returns a transform: PIL image -> tensor [3, size, size].
    """
    return T.Compose([
        T.Resize((size, size)),  # squash to size x size (aspect ratio is not kept)
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
