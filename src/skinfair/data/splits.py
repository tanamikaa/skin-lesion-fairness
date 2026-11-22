"""Lesion-level stratified splits for HAM10000 (prevents leakage). Owner: Afzal Raza."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def lesion_level_split(meta: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """Split so no `lesion_id` appears in more than one split, stratified by `dx`.

    Fractions are approximate (StratifiedGroupKFold works in whole folds of 1/round(1/frac)).
    Returns (train_df, val_df, test_df).

    Why: HAM10000 has several photos of the same lesion. If they landed in different splits,
    the model could "recognise" a lesion it saw in training and test scores would be too high.
    With the defaults: test = 1 of 7 folds (~14.3%), val = 1 of 6 folds of the rest (~14.3%).
    """
    # round(1 / 0.15) = 7 folds; groups=lesion_id keeps all images of a lesion together
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    # next(...) takes only the first fold: its held-out part becomes the test set
    trainval_idx, test_idx = next(sgkf.split(meta, meta["dx"], groups=meta["lesion_id"]))
    trainval, test = meta.iloc[trainval_idx], meta.iloc[test_idx]

    # round(0.85 / 0.15) = 6 folds on the remaining data; first fold's held-out part = val
    sgkf = StratifiedGroupKFold(n_splits=round((1 - test_frac) / val_frac), shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
    # positions from split() are relative to `trainval`, hence trainval.iloc
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Raise if any lesion_id or image_id is shared between two splits.

    Inputs: the three split DataFrames. Returns nothing if all is fine.
    """
    splits = {"train": train, "val": val, "test": test}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:  # each pair once: train-val, train-test, val-test
            for col in ("lesion_id", "image_id"):
                shared = set(splits[a][col]) & set(splits[b][col])  # set intersection
                if shared:
                    raise AssertionError(f"{len(shared)} {col}(s) shared between {a} and {b}")
