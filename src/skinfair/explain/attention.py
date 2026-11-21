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
    """
    store, handles, fused_flags = [], [], []
    for blk in model.blocks:
        fused_flags.append(getattr(blk.attn, "fused_attn", None))
        if fused_flags[-1] is not None:
            blk.attn.fused_attn = False
        handles.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: store.append(o.detach())))
    try:
        yield store
    finally:
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
    """
    was_training = model.training
    model.eval()
    with _capture_attention(model) as attns:
        model(images)
    model.train(was_training)
    if not attns:
        raise RuntimeError("No attention captured - is this a timm ViT with `blocks[i].attn.attn_drop`?")

    B, _, N, _ = attns[0].shape
    eye = torch.eye(N, device=attns[0].device)
    rollout = eye.expand(B, N, N).clone()
    for a in attns:
        a = a.mean(dim=1)  # [B, N, N] average over heads
        if discard_ratio > 0:
            flat = a.reshape(B, -1)
            k = int(flat.shape[1] * discard_ratio)
            low = flat.topk(k, dim=1, largest=False).indices
            flat = flat.scatter(1, low, 0.0)
            a = flat.reshape(B, N, N)  # identity (residual path) is re-added below
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout

    n_prefix = getattr(model, "num_prefix_tokens", 1)
    cls_to_patches = rollout[:, 0, n_prefix:]  # [B, P]
    g = int(math.sqrt(cls_to_patches.shape[1]))
    maps = cls_to_patches.reshape(B, 1, g, g)
    size = out_size or tuple(images.shape[-2:])
    maps = F.interpolate(maps, size=size, mode="bilinear", align_corners=False).squeeze(1)
    flat = maps.reshape(B, -1)
    mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    return ((maps - mn) / (mx - mn + 1e-8)).cpu()


def vit_reshape_transform(tensor, height=14, width=14):
    """Reshape ViT tokens [B, 1+H*W, C] -> [B, C, H, W] (drop CLS) for pytorch-grad-cam."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def gradcam_vit(model, images, targets=None):
    """Grad-CAM on a ViT (target layer: last block's first LayerNorm), for like-for-like comparison."""
    from .gradcam import gradcam

    return gradcam(
        model, images, targets=targets,
        target_layer=model.blocks[-1].norm1,
        reshape_transform=vit_reshape_transform,
    )
