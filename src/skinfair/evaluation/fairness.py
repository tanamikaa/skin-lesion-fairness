"""Per-skin-tone fairness analysis. Owner: Karanpal.

All functions take a predictions DataFrame with columns label, pred, fitzpatrick
(+ image_id for paired comparisons).

IMPORTANT caveat for the report: the Fitzpatrick17k subset only covers the HAM classes that
have a mapping, uses clinical (not dermoscopic) photos, and some FST groups are small
(check the `n` column and the CIs before claiming a gap).
"""
import numpy as np
import pandas as pd
from scipy import stats

from skinfair import CLASSES

from .metrics import classification_metrics, fast_balanced_accuracy, per_class_recall

LIGHT_FST = (1, 2, 3)
DARK_FST = (4, 5, 6)


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
    return {"gap": float(vals.max() - vals.min()), "ratio": float(vals.min() / vals.max()) if vals.max() > 0 else float("nan")}


def light_vs_dark(df: pd.DataFrame, metric="balanced_accuracy"):
    """Fitzpatrick I–III = light, IV–VI = dark."""
    d = df[df["fitzpatrick"] >= 1].copy()
    d["tone"] = np.where(d["fitzpatrick"] <= 3, "light", "dark")
    return per_group_metrics(d, group_col="tone", metric=metric)


def per_class_recall_light_vs_dark(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class recall for light (I-III) vs dark (IV-VI) skin.

    Returns one row per HAM class: class, n_light, n_dark (class support in each group),
    recall_light, recall_dark, gap (light - dark; positive = worse on dark skin).
    Recall is NaN where a group has no samples of that class, so tiny/absent cells are not
    mistaken for 0% recall.
    """
    light, dark = _split_light_dark(df)
    rl = per_class_recall(light["label"], light["pred"])
    rd_ = per_class_recall(dark["label"], dark["pred"])
    nl = np.bincount(light["label"], minlength=len(CLASSES))
    nd = np.bincount(dark["label"], minlength=len(CLASSES))
    out = pd.DataFrame({"class": CLASSES, "n_light": nl, "n_dark": nd, "recall_light": rl, "recall_dark": rd_})
    out["gap"] = out["recall_light"] - out["recall_dark"]
    return out


def _split_light_dark(df):
    d = df[df["fitzpatrick"] >= 1]
    light = d[d["fitzpatrick"].isin(LIGHT_FST)]
    dark = d[d["fitzpatrick"].isin(DARK_FST)]
    if len(light) == 0 or len(dark) == 0:
        raise ValueError("Need rows in both light (I-III) and dark (IV-VI) groups")
    return light, dark


def _ld_gap(light_y, light_p, dark_y, dark_p):
    """light - dark balanced accuracy (positive = worse on dark skin)."""
    return fast_balanced_accuracy(light_y, light_p) - fast_balanced_accuracy(dark_y, dark_p)


def bootstrap_gap_ci(df, n_boot=1000, metric="balanced_accuracy", seed=0, alpha=0.05):
    """Bootstrap CI of the light-vs-dark gap (light - dark; positive = worse on dark skin).

    Each group is resampled independently with replacement. Only balanced accuracy is supported
    (fast implementation). Returns dict(gap, ci_low, ci_high, n_light, n_dark, n_boot).
    """
    if metric != "balanced_accuracy":
        raise NotImplementedError("bootstrap_gap_ci supports metric='balanced_accuracy' only")
    light, dark = _split_light_dark(df)
    ly, lp = light["label"].to_numpy(), light["pred"].to_numpy()
    dy, dp = dark["label"].to_numpy(), dark["pred"].to_numpy()
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        li = rng.integers(0, len(ly), len(ly))
        di = rng.integers(0, len(dy), len(dy))
        gaps[b] = _ld_gap(ly[li], lp[li], dy[di], dp[di])
    lo, hi = np.percentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "gap": float(_ld_gap(ly, lp, dy, dp)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_light": int(len(ly)),
        "n_dark": int(len(dy)),
        "n_boot": n_boot,
    }


def bootstrap_group_ci(df, group_col="fitzpatrick", n_boot=1000, seed=0, alpha=0.05):
    """Per-group balanced accuracy with bootstrap CI. Returns DataFrame[group, n, value, ci_low, ci_high]."""
    if group_col == "fitzpatrick":
        df = df[df[group_col] >= 1]
    rng = np.random.default_rng(seed)
    rows = []
    for g, sub in df.groupby(group_col):
        y, p = sub["label"].to_numpy(), sub["pred"].to_numpy()
        boots = np.array([
            fast_balanced_accuracy(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
        ])
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({group_col: g, "n": len(y), "value": fast_balanced_accuracy(y, p),
                     "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def compare_gap_bootstrap(df_a, df_b, n_boot=2000, seed=0, alpha=0.05):
    """Paired bootstrap test: is model A's light-vs-dark gap different from model B's?

    df_a / df_b are prediction tables of the two models on the *same* images (matched by image_id).
    Images are resampled jointly so the correlation between the models is preserved.
    Returns dict(gap_a, gap_b, diff = gap_a - gap_b, ci_low, ci_high, p_value (two-sided), n_matched).
    diff < 0 means A is fairer (smaller gap).
    """
    m = df_a[["image_id", "label", "pred", "fitzpatrick"]].merge(
        df_b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    m = m[m["fitzpatrick"] >= 1]
    light = m[m["fitzpatrick"].isin(LIGHT_FST)]
    dark = m[m["fitzpatrick"].isin(DARK_FST)]
    if len(light) == 0 or len(dark) == 0:
        raise ValueError("Need matched rows in both light and dark groups")

    def arr(d):
        return d["label"].to_numpy(), d["pred_a"].to_numpy(), d["pred_b"].to_numpy()

    ly, lpa, lpb = arr(light)
    dy, dpa, dpb = arr(dark)
    gap_a, gap_b = _ld_gap(ly, lpa, dy, dpa), _ld_gap(ly, lpb, dy, dpb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        li = rng.integers(0, len(ly), len(ly))
        di = rng.integers(0, len(dy), len(dy))
        diffs[b] = _ld_gap(ly[li], lpa[li], dy[di], dpa[di]) - _ld_gap(ly[li], lpb[li], dy[di], dpb[di])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value for H0: diff == 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "gap_a": float(gap_a), "gap_b": float(gap_b), "diff": float(gap_a - gap_b),
        "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0)),
        "n_matched": int(len(m)),
    }


def mcnemar_test(df_a, df_b):
    """Exact McNemar test on per-image correctness of two models (same images, matched by image_id).

    Returns dict(a_only_correct, b_only_correct, p_value).
    """
    m = df_a[["image_id", "label", "pred"]].merge(df_b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"))
    ca, cb = m["pred_a"] == m["label"], m["pred_b"] == m["label"]
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}
