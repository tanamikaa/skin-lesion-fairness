"""Shared training loop for CNN and ViT.

Owners: Anamika Kumari (loop), Khushi Narwariya (ViT schedule)

TODO(Anamika): AMP, early stopping on val balanced accuracy, best-checkpoint saving, CSV logging.
TODO(Khushi): warm-up + cosine scheduler, layer-decay optimizer when the model is a ViT.
"""


def train(cfg):
    raise NotImplementedError
