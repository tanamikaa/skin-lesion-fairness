"""Lesion-level stratified splits for HAM10000 (prevents leakage). Owner: Afzal Raza."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def lesion_level_split(meta: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """Split so no `lesion_id` appears in more than one split, stratified by `dx`.

    Returns (train_df, val_df, test_df).
    """
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(meta, meta["dx"], groups=meta["lesion_id"]))
    trainval, test = meta.iloc[trainval_idx], meta.iloc[test_idx]

    sgkf = StratifiedGroupKFold(n_splits=round((1 - test_frac) / val_frac), shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test
