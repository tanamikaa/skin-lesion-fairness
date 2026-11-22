# This file makes attention-rollout heat-maps for the Vision Transformer (ViT-B/16).
# Short ViT background:
#   - The 224x224 image is cut into 16x16 patches, so 14 x 14 = 196 patches.
#   - Each patch becomes a "token". One extra CLS token is added in front; its final value predicts the class.
#   - Each transformer block has an attention table saying how much each token looks at every other token.
#   - Attention rollout multiplies the tables of ALL blocks, to see how much the CLS token depends on each patch.
# It also gives the reshape step needed to run Grad-CAM on a ViT, so CNN and ViT can be
# compared with the same method.
# It is part of the "explain" step. scripts/explain.py calls attention_rollout and gradcam_vit.

# math is used for a square root.
import math
# contextmanager lets us write a "with" helper using yield.
from contextlib import contextmanager

# torch is used for tensors.
import torch
# F gives functions like resizing.
import torch.nn.functional as F


# _capture_attention: a "with" helper that temporarily saves every block's attention matrix.
# Input: model: a timm ViT (has `model.blocks`, each with `.attn.attn_drop`).
# Yields: a list `store`; after one forward pass it holds one tensor per block,
#   each [B, heads, N, N] (N = 197 tokens = 1 CLS + 196 patches).
# On exit the hooks are removed and the original settings are restored.
# Why it exists: the model normally throws attention tables away, but rollout needs them.
# A hook is a small function PyTorch calls automatically when a layer finishes running.
# The decorator below lets us use this function with "with".
@contextmanager
def _capture_attention(model):
    """A helper context manager that records attention maps from a ViT.
    Steps: for every block, remember its fused_attn flag and turn it off,
    so the real attention matrix is computed, then add a hook to save it.
    On exit the hooks are removed and the flags are restored.
    Input: model, a timm Vision Transformer.
    Yields a list that fills with one attention tensor per block.
    Needed because fast fused attention never exposes the attention weights.
    """
    # store: saved matrices; handles: hooks to remove later; fused_flags: old fused_attn values.
    store, handles, fused_flags = [], [], []
    # Visit every transformer block.
    for blk in model.blocks:
        # Remember the old fused_attn setting (None if the block has no such flag).
        fused_flags.append(getattr(blk.attn, "fused_attn", None))
        # timm's "fused attention" fast path never builds the full table, so nothing could be captured.
        if fused_flags[-1] is not None:
            # Turn the fast path off while we record.
            blk.attn.fused_attn = False
        # Add a hook on attn_drop (its output is the attention table); the lambda saves a detached
        # copy (values only, no gradient history) each time it runs.
        handles.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: store.append(o.detach())))
    # Use try/finally so cleanup always happens.
    try:
        # Hand the store to the "with" body; the model runs there.
        yield store
    # Always clean up, even if an error happened, so the model is not left changed.
    finally:
        # Remove all hooks.
        for h in handles:
            # Remove one hook.
            h.remove()
        # Put back each block's original fused_attn setting.
        for blk, flag in zip(model.blocks, fused_flags):
            # Only restore if the flag existed.
            if flag is not None:
                # Restore the old value.
                blk.attn.fused_attn = flag


# attention_rollout: makes attention-rollout heat-maps for a timm ViT.
# It follows how the CLS token collects information from image patches across all layers.
# Inputs:
#   model: timm ViT-B/16 (trained classifier).
#   images: Tensor[B, 3, 224, 224], normalised.
#   discard_ratio: share of the weakest attention links set to 0 to reduce noise.
#   out_size: (H, W) of the maps (default: the image size).
# Returns: float Tensor[B, H, W] in [0, 1] on the CPU.
# Why it exists: it is the ViT counterpart of Grad-CAM, so the two models can be compared.
# The decorator below turns off gradients (we only read attention), saving time and memory.
@torch.no_grad()
def attention_rollout(model, images, discard_ratio=0.9, out_size=None):
    """Make a heatmap of where a ViT looked, using attention rollout.
    Steps: run the model in eval mode while capturing attention of all blocks,
    drop the weakest links (discard_ratio), add identity, multiply layers together,
    take the class-token row, reshape to a grid and resize.
    Inputs: model, images, discard_ratio and optional out_size.
    Returns heatmaps scaled to 0-1, one per image.
    Needed to explain which image parts the ViT used.
    """
    # Remember the mode.
    was_training = model.training
    # Switch to eval for stable results (no random dropout).
    model.eval()
    # Run the model once just to fill `attns` with attention matrices.
    with _capture_attention(model) as attns:
        # The output is not used; we only want the captured attention.
        model(images)
    # Restore the original mode.
    model.train(was_training)
    # If nothing was captured, this is not the kind of ViT we expect.
    if not attns:
        # Stop with a clear error message.
        raise RuntimeError("No attention captured - is this a timm ViT with `blocks[i].attn.attn_drop`?")

    # Read sizes: batch, heads (ignored), tokens, tokens.
    B, _, N, _ = attns[0].shape
    # Identity matrix (1 on the diagonal); adding it models the skip connection (a token also keeps itself).
    eye = torch.eye(N, device=attns[0].device)
    # Start the rollout as the identity for every image (before any layer, tokens only see themselves).
    rollout = eye.expand(B, N, N).clone()
    # Go through each layer's attention, from the first block to the last.
    for a in attns:
        # Average over the heads (a head is one of several parallel attention tables).
        a = a.mean(dim=1)
        # Optionally throw away the weakest links so faint noise does not spread through layers.
        if discard_ratio > 0:
            # Put all weights of one image in one row.
            flat = a.reshape(B, -1)
            # How many weights to drop.
            k = int(flat.shape[1] * discard_ratio)
            # Positions of the k smallest weights.
            low = flat.topk(k, dim=1, largest=False).indices
            # Set those positions to 0.
            flat = flat.scatter(1, low, 0.0)
            # Go back to the [B, N, N] shape.
            a = flat.reshape(B, N, N)
        # Add the skip connection: each token also keeps itself.
        a = a + eye
        # Make each row sum to 1 again (adding the identity broke that).
        a = a / a.sum(dim=-1, keepdim=True)
        # Chain this layer onto the earlier layers (matrix multiplication = flow through layers).
        rollout = a @ rollout

    # Number of special tokens (like CLS) before the patch tokens; here it is 1.
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    # Row 0 is the CLS token; skip the special tokens to keep only patches. Shape [B, P], P = 196.
    cls_to_patches = rollout[:, 0, n_prefix:]
    # Side length of the patch grid, e.g. sqrt(196) = 14.
    g = int(math.sqrt(cls_to_patches.shape[1]))
    # Reshape into a grid; the extra channel dimension is needed by interpolate.
    maps = cls_to_patches.reshape(B, 1, g, g)
    # Use the given size, or the image size.
    size = out_size or tuple(images.shape[-2:])
    # Enlarge the small 14x14 grid to the image size, smoothly (bilinear blends neighbours).
    maps = F.interpolate(maps, size=size, mode="bilinear", align_corners=False).squeeze(1)
    # One row per image so we can find min and max.
    flat = maps.reshape(B, -1)
    # Per-image min and max, shaped [B, 1, 1] so they match [B, H, W].
    mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    # Min-max normalise to [0, 1] (the tiny 1e-8 avoids dividing by zero) and move to CPU.
    return ((maps - mn) / (mx - mn + 1e-8)).cpu()


# vit_reshape_transform: turns ViT tokens [B, 1+H*W, C] into a CNN-like grid [B, C, H, W].
# Inputs:
#   tensor: layer output, one vector of C numbers per token (token 0 is CLS).
#   height, width: size of the patch grid (14 x 14).
# Returns: Tensor[B, C, height, width] with the CLS token dropped.
# Why it exists: Grad-CAM was made for CNNs whose layers are 2-D grids; a ViT layer is a flat
# list of tokens, so we re-arrange it into a grid to make Grad-CAM work.
def vit_reshape_transform(tensor, height=14, width=14):
    """Reshape ViT tokens into a picture-like grid for Grad-CAM.
    Steps: drop the first (class) token, reshape the rest into height x width,
    then move the channel axis to the front.
    Inputs: tensor [B, tokens, C], plus height and width of the patch grid (14 x 14).
    Returns a tensor [B, C, height, width].
    Needed because Grad-CAM expects CNN-style feature maps, not token lists.
    """
    # Drop the CLS token (index 0), then lay the patch tokens out on a grid.
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Move the channels first: [B, C, H, W], and return.
    return result.permute(0, 3, 1, 2)


# gradcam_vit: runs Grad-CAM on a ViT so it can be compared with the CNN.
# Inputs:
#   model: timm ViT.
#   images: Tensor[B, 3, 224, 224].
#   targets: classes to explain (default: the predicted class).
# Returns: Tensor[B, H, W] in [0, 1].
# Why it exists: a like-for-like comparison with ResNet Grad-CAM (same method, different network).
def gradcam_vit(model, images, targets=None):
    """Run Grad-CAM on a ViT model.
    Steps: import the general gradcam function, point it at the last block's norm1 layer,
    and give it vit_reshape_transform so tokens become a grid.
    Inputs: model, images and optional targets (class ids).
    Returns the heatmaps that gradcam produces.
    Needed because ViTs have no conv layer, so special settings are required.
    """
    # Imported here (inside the function) to avoid importing it at module load time.
    from .gradcam import gradcam

    # Look at the last block's first LayerNorm (norm1), and use the reshape step above to give it a grid shape.
    return gradcam(
        model, images, targets=targets,
        target_layer=model.blocks[-1].norm1,
        reshape_transform=vit_reshape_transform,
    )
