"""Run a model over a DataLoader and return the predictions table.

Columns (the team interface): image_id, label, pred, prob_0..prob_6, fitzpatrick.
"""
import numpy as np
import pandas as pd
import torch

from skinfair import NUM_CLASSES

PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]


@torch.no_grad()
def predict(model, loader, device=None) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        df = pd.DataFrame(probs, columns=PROB_COLS)
        df.insert(0, "image_id", list(batch["image_id"]))
        df.insert(1, "label", batch["label"].numpy())
        df.insert(2, "pred", probs.argmax(1))
        df["fitzpatrick"] = batch["fitzpatrick"].numpy()
        rows.append(df)
    model.train(was_training)
    return pd.concat(rows, ignore_index=True)


def probs_of(df: pd.DataFrame) -> np.ndarray:
    return df[PROB_COLS].to_numpy()
