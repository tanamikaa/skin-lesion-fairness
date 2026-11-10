"""Explanation-quality metrics. Owner: Karanpal.

TODO(Karanpal): deletion / insertion AUC (Petsiuk et al., RISE 2018);
compare Grad-CAM vs attention maps, overall and per skin-tone group.
"""


def deletion_auc(model, images, heatmaps, steps=20):
    raise NotImplementedError


def insertion_auc(model, images, heatmaps, steps=20):
    raise NotImplementedError
