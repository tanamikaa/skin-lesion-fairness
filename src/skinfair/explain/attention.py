"""Attention rollout for ViT (Abnar & Zuidema, 2020). Owner: Khushi Narwariya.

TODO(Khushi): hook attention matrices from every block, roll out with the residual,
take the CLS row -> 14x14, upsample to 224x224, normalise to [0, 1].
"""


def attention_rollout(model, images, discard_ratio=0.9):
    raise NotImplementedError
