"""Run a model over a DataLoader and return the predictions table.

Columns (the team interface): image_id, label, pred, prob_0..prob_6, fitzpatrick.
"""
import numpy as np
import pandas as pd
import torch

from skinfair import NUM_CLASSES

# Column names for the class probabilities: "prob_0", "prob_1", ... one per class.
PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]


@torch.no_grad()  # inference only: no gradients are tracked (faster, less memory)
def predict(model, loader, device=None) -> pd.DataFrame:
    """Run `model` on every batch of `loader` and collect the results in one table.

    Inputs: a PyTorch model, a DataLoader whose batches are dicts with keys
    image, image_id, label, fitzpatrick, and an optional device (defaults to the
    device the model's weights are on).
    Returns: a DataFrame with one row per image: image_id, label, pred (predicted
    class = highest probability), prob_0..prob_6, fitzpatrick.
    The model's train/eval mode is restored afterwards.
    """
    device = device or next(model.parameters()).device  # use the device of the first weight
    was_training = model.training  # remember the mode so it can be restored at the end
    model.eval()  # eval mode: disables dropout, uses fixed batch-norm statistics
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))  # [B, NUM_CLASSES] raw scores
        # softmax turns logits into probabilities that sum to 1 per image; .float() avoids
        # half-precision issues; move to CPU numpy for pandas.
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        df = pd.DataFrame(probs, columns=PROB_COLS)
        df.insert(0, "image_id", list(batch["image_id"]))  # put id/label/pred as first columns
        df.insert(1, "label", batch["label"].numpy())
        df.insert(2, "pred", probs.argmax(1))  # predicted class = index of the largest probability
        df["fitzpatrick"] = batch["fitzpatrick"].numpy()  # skin type 1-6 (-1 = unknown)
        rows.append(df)
    model.train(was_training)  # restore the original train/eval mode
    return pd.concat(rows, ignore_index=True)  # stack all batch tables into one


def probs_of(df: pd.DataFrame) -> np.ndarray:
    """Return the probability columns of a predictions table as a NumPy array [N, NUM_CLASSES]."""
    return df[PROB_COLS].to_numpy()
