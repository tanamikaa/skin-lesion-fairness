"""Attention rollout for ViT (Abnar & Zuidema, 2020). Owner: Khushi Narwariya.

Also provides the reshape transform needed to run Grad-CAM on a ViT, so CNN and ViT can be
compared with the same method as well as with their native explanations.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def _capture_attention(model):
    """Temporarily make every block expose its softmax attention matrix.

    timm uses fused scaled-dot-product attention by default, which never materialises the
    matrix, so we switch it off and hook `attn_drop` (it receives the post-softmax matrix).

    Used as `with _capture_attention(model) as store: model(x)`. After the forward pass,
    `store` holds one Tensor[B, heads, N, N] per block (N = number of tokens).
    On exit, hooks are removed and the original fused_attn settings are restored.
    """
    # store: captured matrices; handles: hook handles to remove later;
    # fused_flags: original fused_attn value per block (None if the block has no such flag).
    store, handles, fused_flags = [], [], []
    for blk in model.blocks:
        fused_flags.append(getattr(blk.attn, "fused_attn", None))
        if fused_flags[-1] is not None:
            blk.attn.fused_attn = False  # use the explicit (non-fused) path that builds the matrix
        # Forward hook: called after attn_drop runs; o = its output (the attention matrix).
        # detach() drops it from the autograd graph so we only keep the values.
        handles.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: store.append(o.detach())))
    try:
        yield store  # the body of the `with` block runs here
    finally:  # always clean up, even if the forward pass raised an error
        for h in handles:
            h.remove()
        for blk, flag in zip(model.blocks, fused_flags):
            if flag is not None:
                blk.attn.fused_attn = flag


@torch.no_grad()
def attention_rollout(model, images, discard_ratio=0.9, out_size=None):
    """Attention rollout heat-maps for a timm ViT.

    Per block: average heads, drop the lowest `discard_ratio` fraction of attention weights,
    add the identity (residual connection), re-normalise rows, and multiply across blocks.
    The CLS-token row over the patch tokens is reshaped to the patch grid (14x14 for ViT-B/16),
    bilinearly upsampled to the input size and min-max normalised to [0, 1] per image.

    Returns Tensor[B, H, W] on CPU.

    Inputs: a timm ViT, normalised images Tensor[B, 3, H, W], discard_ratio (fraction of
    weakest attention links set to 0 to reduce noise), and optional out_size (H, W) of the maps
    (default: image size). Bright pixels = regions the CLS token (used for classification)
    gathered most information from, across all layers.
    """
    was_training = model.training
    model.eval()
    with _capture_attention(model) as attns:
        model(images)  # forward pass only to fill `attns`; the output is not needed
    model.train(was_training)
    if not attns:
        raise RuntimeError("No attention captured - is this a timm ViT with `blocks[i].attn.attn_drop`?")

    B, _, N, _ = attns[0].shape  # [batch, heads, tokens, tokens]
    eye = torch.eye(N, device=attns[0].device)  # identity matrix [N, N]
    rollout = eye.expand(B, N, N).clone()  # start with identity for every image
    for a in attns:
        a = a.mean(dim=1)  # [B, N, N] average over heads
        if discard_ratio > 0:
            flat = a.reshape(B, -1)  # [B, N*N] all weights of one image in one row
            k = int(flat.shape[1] * discard_ratio)  # how many weights to drop
            low = flat.topk(k, dim=1, largest=False).indices  # positions of the k smallest
            flat = flat.scatter(1, low, 0.0)  # set those positions to 0
            a = flat.reshape(B, N, N)  # identity (residual path) is re-added below
        a = a + eye  # account for the skip connection: each token also keeps itself
        a = a / a.sum(dim=-1, keepdim=True)  # rows sum to 1 again
        rollout = a @ rollout  # chain this layer's attention onto the earlier layers

    n_prefix = getattr(model, "num_prefix_tokens", 1)  # CLS (+ any extra) tokens before patches
    cls_to_patches = rollout[:, 0, n_prefix:]  # [B, P]
    g = int(math.sqrt(cls_to_patches.shape[1]))  # patch grid side, e.g. sqrt(196) = 14
    maps = cls_to_patches.reshape(B, 1, g, g)  # [B, 1, g, g]; channel dim needed by interpolate
    size = out_size or tuple(images.shape[-2:])
    maps = F.interpolate(maps, size=size, mode="bilinear", align_corners=False).squeeze(1)
    flat = maps.reshape(B, -1)
    # Per-image min and max, shaped [B, 1, 1] so they broadcast over [B, H, W].
    mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    return ((maps - mn) / (mx - mn + 1e-8)).cpu()  # scale to [0, 1]; 1e-8 avoids divide by 0


def vit_reshape_transform(tensor, height=14, width=14):
    """Reshape ViT tokens [B, 1+H*W, C] -> [B, C, H, W] (drop CLS) for pytorch-grad-cam.

    Grad-CAM expects CNN-like feature maps, so the patch tokens are laid back out on
    their 14x14 grid (default for ViT-B/16 at 224x224 input).
    """
    # Drop token 0 (CLS), then reshape the sequence of patch tokens into a grid [B, H, W, C].
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)  # move channels first: [B, C, H, W]


def gradcam_vit(model, images, targets=None):
    """Grad-CAM on a ViT (target layer: last block's first LayerNorm), for like-for-like comparison.

    Inputs: a timm ViT, normalised images [B, 3, H, W], optional target classes.
    Returns Tensor[B, H, W] heat-maps in [0, 1] (see gradcam.gradcam).
    """
    from .gradcam import gradcam  # imported here, inside the function

    return gradcam(
        model, images, targets=targets,
        target_layer=model.blocks[-1].norm1,
        reshape_transform=vit_reshape_transform,
    )
