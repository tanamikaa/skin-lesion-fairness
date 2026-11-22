# This file defines how a picture is turned into model input.
# Models need tensors (grids of numbers) of one fixed size with scaled values.
# Training uses random changes (augmentation) so the model sees more variety.
# Validation and test use a fixed pipeline so results are the same every time.

# torchvision transforms are ready-made image steps
from torchvision import transforms as T

# average value of the R, G, B channels on ImageNet; pretrained models expect this scaling
IMAGENET_MEAN = (0.485, 0.456, 0.406)
# spread (standard deviation) of the R, G, B channels on ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)


# Builds the random augmentation pipeline used on training images.
# Args: size: side length in pixels of the output image (default 224).
# Returns: a transform; give it a PIL image and get a normalised tensor [3, size, size].
# Why: random changes each epoch help the model learn general patterns, not memorise.
def train_transforms(size=224):
    """Builds the random augmentation steps used on training images.
    The steps are: random crop and resize, random flips and a small rotation.
    Then a small colour change (hue is kept so skin tone stays the same).
    Last it turns the image into a tensor and normalises it with ImageNet values.
    Input: size, the output side length in pixels (default 224).
    Output: a transform that gives a [3, size, size] tensor. It helps the model generalise.
    """
    # chain the steps one after another
    return T.Compose([
        # crop a random 70-100% part of the image and resize it to size x size
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        # randomly flip left-right
        T.RandomHorizontalFlip(),
        # randomly flip up-down
        T.RandomVerticalFlip(),
        # randomly rotate by up to 20 degrees either way
        T.RandomRotation(20),
        # slightly change brightness, contrast and saturation; hue is 0 so skin tone is kept
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        # turn the image into a tensor with values between 0 and 1
        T.ToTensor(),
        # subtract the mean and divide by the std for each channel
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Builds the fixed (non-random) pipeline for validation, test and Fitzpatrick images.
# Args: size: side length in pixels of the output image (default 224).
# Returns: a transform giving a normalised tensor [3, size, size].
# Why: evaluation must give the same answer every time, so there is no randomness.
def eval_transforms(size=224):
    """Builds the fixed (not random) steps used for validation and test images.
    Step 1: resize the image directly to size x size.
    Step 2: turn it into a tensor with values from 0 to 1.
    Step 3: normalise it with the same ImageNet mean and std as training.
    Input: size, the output side length in pixels (default 224).
    Output: a transform giving a tensor. It has no randomness, so results repeat exactly.
    """
    # chain the steps one after another
    return T.Compose([
        # resize directly to size x size
        T.Resize((size, size)),
        # turn the image into a tensor with values between 0 and 1
        T.ToTensor(),
        # same normalisation as in training
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
