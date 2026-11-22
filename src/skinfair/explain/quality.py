# This file gives numbers that test whether a heat-map is TRUE to what the model uses.
# Methods: deletion / insertion AUC (RISE, 2018), plus pointing-game and IoU against a lesion mask.
# Images are model inputs [B, 3, H, W]; heat-maps are [B, H, W] in [0, 1].
# The score tracked is the model's probability for the target class.
#   - deletion: remove important pixels first, probability should drop fast, so LOWER AUC is better.
#   - insertion: show important pixels first, probability should rise fast, so HIGHER AUC is better.
# AUC = area under the curve of probability against the share of pixels changed (0 to 1).
# Baseline image = the "empty" picture pasted over removed pixels (deletion) or started from (insertion).
# Pointing game and mask IoU are implemented but NOT run in this project (no lesion masks).
# It is part of the "explain" step. scripts/explain.py calls deletion_auc and insertion_auc.

# numpy, pandas and torch are used for arrays, tables and the model.
import numpy as np
import pandas as pd
import torch
# F gives resizing and blur functions.
import torch.nn.functional as F


# _ranks: gives each pixel a rank per image (0 = most important).
# Input: heatmaps: Tensor[B, H, W].
# Returns: LongTensor[B, H*W] of ranks.
# Why it exists: to delete "the top 10% pixels" we need to know which pixels those are.
def _ranks(heatmaps):
    """Give each pixel a rank by heatmap importance.
    Steps: flatten each heatmap, sort its pixels from high to low,
    then scatter positions back so each pixel holds its rank (0 is most important).
    Input: heatmaps tensor [B, H, W].
    Returns an integer tensor with the same layout of ranks.
    Needed so pixels can be deleted or inserted in order of importance.
    """
    # Flatten each map into one long row of pixels.
    flat = heatmaps.flatten(1)
    # order[b, r] = pixel index with rank r (hottest first; stable keeps ties in order).
    order = flat.argsort(dim=1, descending=True, stable=True)
    # Empty tensor with the same shape, to fill with ranks.
    ranks = torch.empty_like(order)
    # Reverse the lookup so each pixel gets its own rank: ranks[b, pixel] = position in `order`.
    ranks.scatter_(1, order, torch.arange(flat.shape[1], device=flat.device).expand_as(order))
    # Give back the ranks.
    return ranks


# _curve: builds the deletion or insertion curve for each image.
# At steps+1 fractions (0 to 1), that share of top pixels is removed (deletion) or
# shown (insertion), and the target-class probability is recorded.
# Inputs:
#   model: trained classifier.
#   images: Tensor[B, 3, H, W].
#   heatmaps: Tensor[B, H, W].
#   steps: number of equal steps between 0 and 1.
#   targets: class per image (default: the prediction on the full image).
#   mode: "deletion" or "insertion".
# Returns: (fractions, scores of shape [B, steps+1]) as numpy arrays.
# The decorator below turns off gradients (we only run the model forward), so it is faster.
@torch.no_grad()
def _curve(model, images, heatmaps, steps, targets, mode):
    """Build a deletion or insertion curve for a batch of images.
    Steps: rank pixels, then step by step hide (or reveal) more of the most important ones,
    run the model and record the target class probability each time.
    Inputs: model, images, heatmaps, steps, targets and mode ("deletion" or "insertion").
    Returns the fractions used and the score curve per image.
    Needed as the shared base of deletion_auc and insertion_auc.
    """
    # Find the device the model is on.
    device = next(model.parameters()).device
    # Remember the mode.
    was_training = model.training
    # Use eval mode for stable results.
    model.eval()
    # Move the data to the model's device.
    images, heatmaps = images.to(device), heatmaps.to(device)
    # Read the image sizes.
    B, _, H, W = images.shape
    # Resize heat-maps to the image size if they differ.
    if heatmaps.shape[-2:] != (H, W):
        # [:, None] adds a channel dimension and [:, 0] removes it again.
        heatmaps = F.interpolate(heatmaps[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    # Default target: the class predicted on the full image.
    if targets is None:
        # Highest-scoring class per image.
        targets = model(images).argmax(1)
    # Make sure targets is a tensor on the right device.
    targets = torch.as_tensor(targets, device=device)

    # Pixel ranks with a channel dimension so they work across RGB (same rank for all 3 colours).
    ranks = _ranks(heatmaps).reshape(B, 1, H, W)
    # Baseline for deletion: zeros (the average colour after normalising).
    zero = torch.zeros_like(images)
    # Baseline for insertion: a 31x31 box blur of the image; it hides detail but keeps overall colours.
    blur = F.avg_pool2d(images, kernel_size=31, stride=1, padding=15, count_include_pad=False)
    # Number of pixels per image.
    total = H * W
    # Fractions of pixels to change at each step (e.g. 0, 0.05, ..., 1.0 for 20 steps).
    fracs = np.linspace(0, 1, steps + 1)
    # This list will hold the scores from each step.
    scores = []
    # One step for each fraction.
    for f in fracs:
        # How many pixels to change in this step.
        k = int(round(f * total))
        # True for the k most important pixels (rank 0 is the most important).
        top = ranks < k
        # Deletion: replace top pixels with the baseline, keep the rest.
        if mode == "deletion":
            # Pick the zero baseline where `top` is True, otherwise the real image.
            x = torch.where(top, zero, images)
        # Insertion: keep top pixels sharp, leave the rest blurred.
        else:
            # Pick the real image where `top` is True, otherwise the blurred image.
            x = torch.where(top, images, blur)
        # Probability of the target class: softmax over classes, then pick the target column with gather.
        p =torch.softmax(model(x).float(), dim=1).gather(1, targets[:, None]).squeeze(1)
        # Save this step's scores as numpy.
        scores.append(p.cpu().numpy())
    # Put the model back in its original mode.
    model.train(was_training)
    # Stack steps into shape [B, steps+1] and return with the fractions.
    return fracs, np.stack(scores, axis=1)


# deletion_auc: deletion AUC per image (lower is better).
# Removing the marked pixels should hurt the prediction fast.
# Inputs: model, images [B, 3, H, W], heatmaps [B, H, W], steps (curve points), targets (classes).
# Returns: numpy array [B], one score per image.
def deletion_auc(model, images, heatmaps, steps=20, targets=None):
    """Measure how fast the prediction falls when top pixels are removed.
    Steps: build the deletion curve with _curve,
    then find the area under it using the trapezoid rule.
    Inputs: model, images, heatmaps, steps and optional targets.
    Returns one number per image; lower means a more faithful heatmap.
    Needed to check that the heatmap points to truly important pixels.
    """
    # Get the deletion curve.
    fracs, s = _curve(model, images, heatmaps, steps, targets, "deletion")
    # Area under the curve of probability vs fraction removed (trapezoid rule, one row per image).
    return np.trapezoid(s, fracs, axis=1)


# insertion_auc: insertion AUC per image (higher is better).
# Showing the marked pixels should restore the prediction fast.
# Inputs and returns: same as deletion_auc.
def insertion_auc(model, images, heatmaps, steps=20, targets=None):
    """Measure how fast the prediction rises when top pixels are added.
    Steps: build the insertion curve with _curve,
    then find the area under it using the trapezoid rule.
    Inputs: model, images, heatmaps, steps and optional targets.
    Returns one number per image; higher means a more faithful heatmap.
    Needed as the partner test to deletion_auc.
    """
    # Get the insertion curve.
    fracs, s = _curve(model, images, heatmaps, steps, targets, "insertion")
    # Area under the curve of probability vs fraction shown.
    return np.trapezoid(s, fracs, axis=1)


# pointing_game: gives 1.0 if the heat-map's hottest pixel is inside the lesion mask, else 0.0.
# Inputs: heatmaps [B, H, W]; masks [B, H, W] (lesion = 1). Not run in this project (no masks).
# Returns: numpy array [B] of 0.0 / 1.0.
def pointing_game(heatmaps, masks):
    """Check if the hottest heatmap pixel lands inside the lesion mask.
    Steps: align heatmaps and masks to the same size,
    find the index of the maximum of each heatmap, and look up the mask there.
    Inputs: heatmaps and masks (lesion outlines).
    Returns an array of 1.0 (hit) or 0.0 (miss) per image.
    Needed to see if the model looks at the lesion, not the background.
    """
    # Make types match and sizes equal.
    heatmaps, masks = _align(heatmaps, masks)
    # Position of the hottest pixel in each image.
    idx = heatmaps.flatten(1).argmax(1)
    # Read the mask at that pixel and turn True/False into 1.0/0.0.
    return masks.flatten(1).gather(1, idx[:, None]).squeeze(1).float().numpy()


# mask_iou: IoU (overlap divided by combined area) of the hot region and the lesion mask, per image.
# Inputs:
#   heatmaps: [B, H, W].
#   masks: [B, H, W].
#   threshold: heat value counted as "hot".
# Returns: numpy array [B] in [0, 1]; 1 = perfect match. Not run in this project (no masks).
def mask_iou(heatmaps, masks, threshold=0.5):
    """Measure how much a heatmap overlaps the lesion mask (IoU).
    Steps: align sizes, make a binary prediction using the threshold,
    count the overlap (intersection) and the combined area (union), then divide.
    Inputs: heatmaps, masks and a threshold (default 0.5).
    Returns an array with one IoU value per image.
    Needed to score how well the highlighted area matches the lesion.
    """
    # Make types match and sizes equal.
    heatmaps, masks = _align(heatmaps, masks)
    # The "hot region": pixels at or above the threshold.
    pred = heatmaps >= threshold
    # Pixels that are in both the hot region and the mask.
    inter = (pred & masks).flatten(1).sum(1).float()
    # Pixels that are in either one.
    union = (pred | masks).flatten(1).sum(1).float()
    # Divide; clamp avoids 0/0 when both are empty.
    return (inter / union.clamp(min=1)).numpy()


# _align: makes heat-maps float and masks boolean, resizing masks to the heat-map size if needed.
# Inputs: heatmaps and masks (array-likes of shape [B, H, W]).
# Returns: (float heatmaps, boolean masks).
# Why it exists: pointing_game and mask_iou compare the two pixel by pixel, so shapes and types must match.
def _align(heatmaps, masks):
    """Make heatmaps and masks ready to compare.
    Steps: convert both to tensors, and if the mask size differs from the heatmap,
    resize the mask with nearest-neighbour, then make the mask True or False.
    Inputs: heatmaps and masks.
    Returns a pair: float heatmaps and boolean masks of the same size.
    Needed because pointing_game and mask_iou need equal shapes.
    """
    # Heat-maps become float tensors.
    heatmaps = torch.as_tensor(heatmaps).float()
    # Masks become tensors.
    masks = torch.as_tensor(masks)
    # Resize masks only if the sizes differ.
    if masks.shape[-2:] != heatmaps.shape[-2:]:
        # "nearest" keeps mask values 0 or 1 (no blending at the edges).
        masks = F.interpolate(masks[:, None].float(), size=heatmaps.shape[-2:], mode="nearest")[:, 0]
    # Turn masks into True/False and return both.
    return heatmaps, masks > 0.5


# quality_by_group: averages per-image scores for each group (e.g. skin type), with standard error (sem).
# Inputs:
#   scores: one number per image.
#   groups: group id per image (e.g. Fitzpatrick type).
#   method: name of the explanation method (used as a label).
#   metric: name for the mean column.
# Returns: pandas table with columns method, group, n, <metric>, sem. Unknown group (-1) is ignored.
# Why it exists: if quality is much worse for some skin types, the explanations are less
# trustworthy for those patients.
def quality_by_group(scores, groups, method: str, metric: str = "value"):
    """Average explanation scores for each group such as skin tone.
    Steps: build a table of scores and groups, drop rows with group -1,
    group by group to get count, mean and standard error, then add the method name.
    Inputs: scores, groups, method name and a metric column name.
    Returns a DataFrame with method, group, n, metric and sem.
    Needed to compare explanation quality across groups.
    """
    # Put scores and groups side by side in one table.
    df = pd.DataFrame({"score": np.asarray(scores), "group": np.asarray(groups)})
    # Drop images with unknown group.
    df = df[df["group"] != -1]
    # For each group: count, mean and standard error of the mean.
    agg = df.groupby("group")["score"].agg(["count", "mean", "sem"]).reset_index()
    # Give the columns clear names.
    agg.columns = ["group", "n", metric, "sem"]
    # Add the method name as the first column.
    agg.insert(0, "method", method)
    # Give back the table.
    return agg
