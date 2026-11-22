# This file runs a trained model over a whole dataset and records, for every image,
# the true label, the predicted class and the probability of each class.
# It is the first step of "evaluate": the model is already trained and we ask it for answers.
# Its output table feeds metrics.py and fairness.py. scripts/evaluate.py calls it.
# Table columns: image_id, label, pred, prob_0 ... prob_6, fitzpatrick (-1 = skin type unknown).

# numpy and pandas handle arrays and tables.
import numpy as np
import pandas as pd
# torch runs the model.
import torch

# Number of classes, shared by the whole project.
from skinfair import NUM_CLASSES

# Column names for the probabilities: prob_0, prob_1, ... one per class.
PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]


# predict: runs the model on every batch and returns one table with a row per image.
# Inputs:
#   model: a trained torch model that maps images to 7 raw scores (logits).
#   loader: a DataLoader whose batches are dicts with keys image, image_id, label, fitzpatrick.
#   device: "cuda" or "cpu"; if None, the device of the model's weights is used.
# Returns: a pandas table with columns image_id, label, pred, prob_0..prob_6, fitzpatrick.
# Why it exists: every fairness number is computed from this table, so it keeps the skin
# type next to each prediction.
# The decorator below turns off gradient tracking, because we only predict (faster, less memory).
@torch.no_grad()
def predict(model, loader, device=None) -> pd.DataFrame:
    """Run the model over a data loader and collect all predictions.
    Steps: pick the device, switch the model to eval mode, loop over batches
    without gradients, store labels, predictions and probabilities,
    then restore the old training mode.
    Inputs: model, loader and an optional device.
    Returns a pandas DataFrame with one row per image.
    Needed so evaluation and fairness code get one simple table.
    """
    # If no device is given, use the device where the model weights live.
    device = device or next(model.parameters()).device
    # Remember if the model was in train mode so we can put it back later.
    was_training = model.training
    # Eval mode turns off dropout and fixes batch norm, so predictions are stable.
    model.eval()
    # This list will hold one small table per batch.
    rows = []
    # Go through the data one batch at a time.
    for batch in loader:
        # Run the model on the images to get raw scores (logits).
        # non_blocking=True lets the copy to the GPU overlap with other work.
        logits = model(batch["image"].to(device, non_blocking=True))
        # Turn scores into probabilities (softmax makes them positive and sum to 1) and move to CPU numpy.
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        # Make a table with one probability column per class.
        df = pd.DataFrame(probs, columns=PROB_COLS)
        # Put the image id as the first column.
        df.insert(0, "image_id", list(batch["image_id"]))
        # Put the true label as the second column.
        df.insert(1, "label", batch["label"].numpy())
        # Put the predicted class (the highest probability in each row) as the third column.
        df.insert(2, "pred", probs.argmax(1))
        # Add the skin type (1-6, or -1 if unknown) at the end.
        df["fitzpatrick"] = batch["fitzpatrick"].numpy()
        # Keep this batch's table.
        rows.append(df)
    # Put the model back in its original mode.
    model.train(was_training)
    # Join all batch tables into one big table and return it.
    return pd.concat(rows, ignore_index=True)


# probs_of: returns the probability columns of a predictions table as a numpy array.
# Input: the table made by predict (or read back from a predictions CSV).
# Returns: float array of shape [N, NUM_CLASSES], one row per image (used for AUROC).
def probs_of(df: pd.DataFrame) -> np.ndarray:
    """Pull the probability columns out of the predictions table.
    Steps: select the columns named in PROB_COLS
    and convert them to a NumPy array.
    Input: df, the DataFrame made by predict().
    Returns an array of shape [N, 7] with class probabilities.
    Needed because metrics like AUROC want probabilities as a plain array.
    """
    # Pick only the probability columns and convert them to numpy.
    return df[PROB_COLS].to_numpy()
