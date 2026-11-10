"""Per-skin-tone fairness analysis. Owner: Karanpal."""
import numpy as np
import pandas as pd

from .metrics import classification_metrics


def per_group_metrics(df: pd.DataFrame, group_col="fitzpatrick", metric="balanced_accuracy"):
    """df needs columns: label, pred, <group_col>. Fitzpatrick rows with -1 (unknown) are ignored."""
    if group_col == "fitzpatrick":
        df = df[df[group_col] >= 1]
    rows = [
        {group_col: g, "n": len(sub), metric: classification_metrics(sub["label"], sub["pred"])[metric]}
        for g, sub in df.groupby(group_col)
    ]
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def fairness_gap(group_df: pd.DataFrame, metric="balanced_accuracy"):
    vals = group_df[metric].to_numpy()
    return {"gap": float(vals.max() - vals.min()), "ratio": float(vals.min() / vals.max())}


def light_vs_dark(df: pd.DataFrame, metric="balanced_accuracy"):
    """Fitzpatrick I–III = light, IV–VI = dark."""
    d = df[df["fitzpatrick"] >= 1].copy()
    d["tone"] = np.where(d["fitzpatrick"] <= 3, "light", "dark")
    return per_group_metrics(d, group_col="tone", metric=metric)


def bootstrap_gap_ci(df, n_boot=1000, metric="balanced_accuracy", seed=0):
    """TODO(Karanpal): bootstrap 95% CI of the light-vs-dark gap."""
    raise NotImplementedError
