# This file makes Grad-CAM heat-maps (for ResNet-50, and optionally ViT) that show which
# parts of a skin image pushed the model towards a class.
# Grad-CAM idea in 3 steps:
#   1. Run the image through the model and take the score of the class we want to explain.
#   2. Use the gradient (how much the score changes if a number inside the model changes)
#      to find which feature maps of a chosen layer helped that score.
#   3. Add those feature maps up (weighted), keep the positive part, stretch to the image size.
#      Bright = that region raised the class score.
# It is part of the "explain" step. scripts/explain.py (ResNet) and explain/attention.py
# (gradcam_vit) call it.
# Use: gradcam(model, images) -> Tensor[B, H, W] heat-maps in [0, 1].

# torch runs the model and handles tensors.
import torch
# The Grad-CAM tool from the pytorch_grad_cam library (we do not write the maths ourselves).
from pytorch_grad_cam import GradCAM
# Helper that says which class output we want to explain.
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# default_target_layer: picks the layer Grad-CAM should look at for a ResNet-50.
# Input: model: a torchvision-style ResNet (has a `layer4` attribute).
# Returns: the last residual block of the last stage (a torch module).
# Why it exists: deep layers know "what" is in the image, and the last conv layer still keeps
# a rough 7x7 layout ("where"), so it is the best place to ask where the model looked.
def default_target_layer(model):
    """Choose the layer that Grad-CAM looks at for a ResNet.
    Steps: take the model's last stage (layer4)
    and return its last block.
    Input: model, a ResNet.
    Returns that layer module.
    Needed because Grad-CAM needs one convolution layer
    that still keeps spatial information.
    """
    # layer4 is the last stage; [-1] is its last block.
    return model.layer4[-1]


# gradcam: makes Grad-CAM heat-maps, one per image, scaled to [0, 1].
# Inputs:
#   model: trained classifier (torch module).
#   images: Tensor[B, 3, H, W], normalised, on the same device as the model.
#   targets: class numbers to explain, one per image (default: the predicted class).
#   target_layer: layer to look at (default: last ResNet block).
#   reshape_transform: function turning the layer output into [B, C, h, w]; needed for ViT.
# Returns: float Tensor[B, H, W] in [0, 1] (0 = unimportant, 1 = most important).
# Why it exists: this is the main "why did the model say that?" picture of the project.
def gradcam(model, images, targets=None, target_layer=None, reshape_transform=None):
    """Make Grad-CAM heatmaps showing which pixels drove a prediction.
    Steps: pick the target layer, set eval mode, find target classes if not given,
    run the Grad-CAM engine, then restore the training mode.
    Inputs: model, images, targets, target_layer and reshape_transform (all optional but images).
    Returns heatmaps scaled to 0-1, one per image.
    Needed to explain what the model focused on.
    """
    # Use the given layer, or fall back to the ResNet default.
    target_layer = target_layer if target_layer is not None else default_target_layer(model)
    # Remember the mode (train or eval) so we can restore it later.
    was_training = model.training
    # Eval mode switches off dropout and uses fixed batch-norm values, so output is repeatable.
    model.eval()
    # If no classes were given, explain what the model predicts.
    if targets is None:
        # No gradients are needed just to get predictions (saves memory and time).
        with torch.no_grad():
            # argmax(1) = index of the highest score per image = the predicted class.
            targets = model(images).argmax(1)
    # Tell the library which class output to explain for each image (int(t) makes a plain number).
    cam_targets = [ClassifierOutputTarget(int(t)) for t in targets]
    # Set up Grad-CAM on the chosen layer; "with" makes sure the library's hooks are removed at the end.
    with GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        # Turn gradients on even if the caller turned them off (Grad-CAM needs them).
        with torch.enable_grad():
            # Compute the maps; the result is a numpy array [B, H, W].
            maps = cam(input_tensor=images, targets=cam_targets)
    # Put the model back in its original mode.
    model.train(was_training)
    # Convert to a float tensor and clip to [0, 1] to fix tiny rounding errors (clamp_ works in place).
    return torch.as_tensor(maps, dtype=torch.float32).clamp_(0, 1)
