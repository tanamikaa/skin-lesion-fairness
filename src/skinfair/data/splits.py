# This file splits the HAM10000 table into train, validation and test parts.
# The split is done per lesion, not per image, because one lesion has several photos.
# If photos of one lesion were in different splits, the test score would be too good
# (this is called data leakage).

# pandas gives us tables (DataFrames)
import pandas as pd
# StratifiedGroupKFold keeps groups together and keeps class shares similar in each fold
from sklearn.model_selection import StratifiedGroupKFold


# Splits the metadata into train, val and test so a lesion never appears in two parts.
# Args:
#   meta: the metadata table (needs columns "dx" and "lesion_id").
#   val_frac: wanted share of validation data (0.15 = 15%).
#   test_frac: wanted share of test data.
#   seed: number that makes the split repeatable.
# Returns: three tables (train, val, test), roughly 70% / 15% / 15%.
# Why: to test the model fairly on lesions it has never seen.
def lesion_level_split(meta: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """Splits the metadata into train, validation and test tables.
    Step 1: take out about test_frac of the data as test, keeping each lesion together.
    Step 2: split the rest into train and validation the same way.
    The split also keeps class shares similar and is repeatable with the seed.
    Input: meta table, val_frac, test_frac and seed.
    Output: three tables (train, val, test). It stops leakage of one lesion across splits.
    """
    # make a splitter with about 1/test_frac folds, so one fold is about the test share
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    # take only the first split: the held-out fold becomes the test set
    trainval_idx, test_idx = next(sgkf.split(meta, meta["dx"], groups=meta["lesion_id"]))
    # use the row positions to get the two tables
    trainval, test = meta.iloc[trainval_idx], meta.iloc[test_idx]

    # make a second splitter on the remaining data; the number of folds is chosen so
    # one fold is about val_frac of the original data
    sgkf = StratifiedGroupKFold(n_splits=round((1 - test_frac) / val_frac), shuffle=True, random_state=seed)
    # split trainval into train and validation (again keeping lesions together)
    train_idx, val_idx = next(sgkf.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
    # positions refer to trainval, so we index trainval here, not meta
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test


# Checks that no lesion or image is shared between two splits.
# Args: train, val, test: the three tables from lesion_level_split.
# Returns: nothing if all is fine; raises AssertionError if something is shared.
# Why: a safety check that proves there is no data leakage.
def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Checks that no lesion or image is shared between two splits.
    Step 1: put the three tables in a dict with names.
    Step 2: compare every pair of splits once.
    Step 3: for both lesion_id and image_id, find ids that appear in both.
    Input: train, val and test tables.
    Output: nothing if fine, but it raises an error if something is shared. It proves no data leakage.
    """
    # put the splits in a dict so each has a name for error messages
    splits = {"train": train, "val": val, "test": test}
    # list of the split names
    names = list(splits)
    # go over each split a
    for i, a in enumerate(names):
        # compare with each later split b, so every pair is checked once
        for b in names[i + 1:]:
            # check both id columns
            for col in ("lesion_id", "image_id"):
                # ids that appear in both splits
                shared = set(splits[a][col]) & set(splits[b][col])
                # any shared id means leakage
                if shared:
                    # stop with a clear message
                    raise AssertionError(f"{len(shared)} {col}(s) shared between {a} and {b}")
