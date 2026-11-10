"""Grad-CAM for ResNet-50. Owner: Anamika Kumari.

TODO(Anamika): use pytorch-grad-cam with target layer model.layer4[-1];
return heat-maps Tensor[B, 224, 224] normalised to [0, 1].
"""


def gradcam(model, images, targets=None):
    raise NotImplementedError
