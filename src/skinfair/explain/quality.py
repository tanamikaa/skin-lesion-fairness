"""Explanation-quality metrics. Owner: Karanpal.

Deletion / insertion AUC (Petsiuk et al., RISE 2018), plus pointing-game and IoU against a
lesion mask when masks are available (HAM10000 segmentation masks are a separate download).

Convention: images are normalised model inputs [B, 3, H, W]; heat-maps are [B, H, W] in [0, 1].
The score tracked is the softmax probability of the target class (default: the model's
prediction on the intact image).

  * deletion  : remove the most-important pixels first -> prob should drop fast  -> LOWER AUC is better
  * insertion : reveal the most-important pixels first  -> prob should rise fast -> HIGHER AUC is better
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _ranks(heatmaps):
    """Per-image pixel rank, 0 = most important. Returns LongTensor[B, H*W]."""
    flat = heatmaps.flatten(1)
    order = flat.argsort(dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(flat.shape[1], device=flat.device).expand_as(order))
    return ranks


@torch.no_grad()
def _curve(model, images, heatmaps, steps, targets, mode):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    images, heatmaps = images.to(device), heatmaps.to(device)
    B, _, H, W = images.shape
    if heatmaps.shape[-2:] != (H, W):
        heatmaps = F.interpolate(heatmaps[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    if targets is None:
        targets = model(images).argmax(1)
    targets = torch.as_tensor(targets, device=device)

    ranks = _ranks(heatmaps).reshape(B, 1, H, W)
    # baseline: zeros == the dataset mean colour after normalisation (deletion) /
    # heavily blurred image (insertion), the usual RISE choices.
    zero = torch.zeros_like(images)
    blur = F.avg_pool2d(images, kernel_size=31, stride=1, padding=15, count_include_pad=False)
    total = H * W
    fracs = np.linspace(0, 1, steps + 1)
    scores = []
    for f in fracs:
        k = int(round(f * total))
        top = ranks < k  # [B, 1, H, W] True for the k most important pixels
        if mode == "deletion":
            x = torch.where(top, zero, images)
        else:
            x = torch.where(top, images, blur)
        p = torch.softmax(model(x).float(), dim=1).gather(1, targets[:, None]).squeeze(1)
        scores.append(p.cpu().numpy())
    model.train(was_training)
    return fracs, np.stack(scores, axis=1)  # [B, steps+1]


def deletion_auc(model, images, heatmaps, steps=20, targets=None):
    """Per-image deletion AUC (lower is better). Returns np.ndarray[B]."""
    fracs, s = _curve(model, images, heatmaps, steps, targets, "deletion")
    return np.trapezoid(s, fracs, axis=1)


def insertion_auc(model, images, heatmaps, steps=20, targets=None):
    """Per-image insertion AUC (higher is better). Returns np.ndarray[B]."""
    fracs, s = _curve(model, images, heatmaps, steps, targets, "insertion")
    return np.trapezoid(s, fracs, axis=1)


def pointing_game(heatmaps, masks):
    """1.0 if the heat-map's arg-max pixel lies inside the lesion mask, else 0.0. Returns np.ndarray[B].

    masks: bool/0-1 Tensor[B, H, W] (resized to the heat-map size if needed, nearest).
    """
    heatmaps, masks = _align(heatmaps, masks)
    idx = heatmaps.flatten(1).argmax(1)
    return masks.flatten(1).gather(1, idx[:, None]).squeeze(1).float().numpy()


def mask_iou(heatmaps, masks, threshold=0.5):
    """IoU between the thresholded heat-map (>= threshold) and the lesion mask. Returns np.ndarray[B]."""
    heatmaps, masks = _align(heatmaps, masks)
    pred = heatmaps >= threshold
    inter = (pred & masks).flatten(1).sum(1).float()
    union = (pred | masks).flatten(1).sum(1).float()
    return (inter / union.clamp(min=1)).numpy()


def _align(heatmaps, masks):
    heatmaps = torch.as_tensor(heatmaps).float()
    masks = torch.as_tensor(masks)
    if masks.shape[-2:] != heatmaps.shape[-2:]:
        masks = F.interpolate(masks[:, None].float(), size=heatmaps.shape[-2:], mode="nearest")[:, 0]
    return heatmaps, masks > 0.5


def quality_by_group(scores, groups, method: str, metric: str = "value"):
    """Aggregate per-image scores by group (e.g. Fitzpatrick type).

    Returns DataFrame[method, group, n, value(mean), sem], ignoring group == -1.
    """
    df = pd.DataFrame({"score": np.asarray(scores), "group": np.asarray(groups)})
    df = df[df["group"] != -1]
    agg = df.groupby("group")["score"].agg(["count", "mean", "sem"]).reset_index()
    agg.columns = ["group", "n", metric, "sem"]
    agg.insert(0, "method", method)
    return agg
