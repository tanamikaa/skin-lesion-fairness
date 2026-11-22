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

# Fitzpatrick skin types (FST) I-III are grouped as "light", IV-VI as "dark".
LIGHT_FST = (1, 2, 3)
DARK_FST = (4, 5, 6)


def per_group_metrics(df: pd.DataFrame, group_col="fitzpatrick", metric="balanced_accuracy"):
    """df needs columns: label, pred, <group_col>. Fitzpatrick rows with -1 (unknown) are ignored.

    Computes one metric (default balanced accuracy) separately for each group, e.g. each
    skin type. Returns a DataFrame with columns <group_col>, n (images in group), <metric>,
    sorted by group. Comparing the rows shows whether the model works equally well per group.
    """
    if group_col == "fitzpatrick":
        df = df[df[group_col] >= 1]  # drop unknown skin type (-1)
    # One row per group: group value, its size, and the metric on that group's rows only.
    rows = [
        {group_col: g, "n": len(sub), metric: classification_metrics(sub["label"], sub["pred"])[metric]}
        for g, sub in df.groupby(group_col)
    ]
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def fairness_gap(group_df: pd.DataFrame, metric="balanced_accuracy"):
    """Summarise how unequal a metric is across groups (output of per_group_metrics).

    gap   = best group - worst group. 0 means all groups are served equally well; a large
            gap means the model is clearly worse for some skin types.
    ratio = worst group / best group. 1.0 = perfectly equal; NaN if the best value is 0.
    Returns dict(gap, ratio).
    """
    vals = group_df[metric].to_numpy()
    return {"gap": float(vals.max() - vals.min()), "ratio": float(vals.min() / vals.max()) if vals.max() > 0 else float("nan")}


def light_vs_dark(df: pd.DataFrame, metric="balanced_accuracy"):
    """Fitzpatrick I–III = light, IV–VI = dark.

    Same as per_group_metrics but with only two groups. Returns a 2-row DataFrame
    with columns tone, n, <metric>.
    """
    d = df[df["fitzpatrick"] >= 1].copy()  # drop unknown skin type; copy so we can add a column
    d["tone"] = np.where(d["fitzpatrick"] <= 3, "light", "dark")  # label each row light/dark
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
    nl = np.bincount(light["label"], minlength=len(CLASSES))  # images per class, light group
    nd = np.bincount(dark["label"], minlength=len(CLASSES))  # images per class, dark group
    out = pd.DataFrame({"class": CLASSES, "n_light": nl, "n_dark": nd, "recall_light": rl, "recall_dark": rd_})
    out["gap"] = out["recall_light"] - out["recall_dark"]
    return out


def _split_light_dark(df):
    """Split a predictions table into (light rows, dark rows), dropping unknown skin type.

    Raises ValueError if either group is empty (a gap cannot be computed then).
    """
    d =df[df["fitzpatrick"] >= 1]
    light = d[d["fitzpatrick"].isin(LIGHT_FST)]
    dark = d[d["fitzpatrick"].isin(DARK_FST)]
    if len(light) == 0 or len(dark) == 0:
        raise ValueError("Need rows in both light (I-III) and dark (IV-VI) groups")
    return light, dark


def _ld_gap(light_y, light_p, dark_y, dark_p):
    """light - dark balanced accuracy (positive = worse on dark skin).

    Inputs: true/predicted labels for the light group and for the dark group. Returns a float.
    """
    return fast_balanced_accuracy(light_y, light_p) - fast_balanced_accuracy(dark_y, dark_p)


def bootstrap_gap_ci(df, n_boot=1000, metric="balanced_accuracy", seed=0, alpha=0.05):
    """Bootstrap CI of the light-vs-dark gap (light - dark; positive = worse on dark skin).

    Each group is resampled independently with replacement. Only balanced accuracy is supported
    (fast implementation). Returns dict(gap, ci_low, ci_high, n_light, n_dark, n_boot).

    Why: with small groups the measured gap can be partly luck. The bootstrap redraws the
    data many times to see how much the gap varies. If the CI [ci_low, ci_high] excludes 0,
    the gap is unlikely to be just noise. alpha=0.05 gives a 95% CI.
    """
    if metric != "balanced_accuracy":
        raise NotImplementedError("bootstrap_gap_ci supports metric='balanced_accuracy' only")
    light, dark = _split_light_dark(df)
    ly, lp = light["label"].to_numpy(), light["pred"].to_numpy()  # light: true, predicted
    dy, dp = dark["label"].to_numpy(), dark["pred"].to_numpy()  # dark: true, predicted
    rng = np.random.default_rng(seed)  # seeded random generator -> reproducible results
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        # Random row indices with replacement, same size as each group.
        li = rng.integers(0, len(ly), len(ly))
        di = rng.integers(0, len(dy), len(dy))
        gaps[b] = _ld_gap(ly[li], lp[li], dy[di], dp[di])
    # Percentile CI: e.g. the 2.5th and 97.5th percentiles of the bootstrap gaps.
    lo, hi = np.percentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "gap": float(_ld_gap(ly, lp, dy, dp)),  # point estimate on the real (not resampled) data
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_light": int(len(ly)),
        "n_dark": int(len(dy)),
        "n_boot": n_boot,
    }


def bootstrap_group_ci(df, group_col="fitzpatrick", n_boot=1000, seed=0, alpha=0.05):
    """Per-group balanced accuracy with bootstrap CI. Returns DataFrame[group, n, value, ci_low, ci_high].

    The group column keeps its name (<group_col>). value = balanced accuracy on the real data;
    ci_low/ci_high = bootstrap percentile CI. Wide CIs mean the group is too small to trust.
    """
    if group_col == "fitzpatrick":
        df = df[df[group_col] >= 1]  # drop unknown skin type (-1)
    rng = np.random.default_rng(seed)
    rows = []
    for g, sub in df.groupby(group_col):
        y, p = sub["label"].to_numpy(), sub["pred"].to_numpy()
        # n_boot resamples (with replacement) of this group's rows; balanced accuracy of each.
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

    Use this to check if e.g. a debiasing method really reduced the skin-tone gap, or if the
    change could be random. A small p_value (e.g. < 0.05) suggests a real difference.
    """
    # Join both tables on image_id: one row per image with pred_a and pred_b.
    # validate="one_to_one" raises an error if an image_id is duplicated.
    m = df_a[["image_id", "label", "pred", "fitzpatrick"]].merge(
        df_b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    m = m[m["fitzpatrick"] >= 1]
    light = m[m["fitzpatrick"].isin(LIGHT_FST)]
    dark = m[m["fitzpatrick"].isin(DARK_FST)]
    if len(light) == 0 or len(dark) == 0:
        raise ValueError("Need matched rows in both light and dark groups")

    def arr(d):
        """Return (true labels, model A preds, model B preds) of a table as NumPy arrays."""
        return d["label"].to_numpy(), d["pred_a"].to_numpy(), d["pred_b"].to_numpy()

    ly, lpa, lpb = arr(light)
    dy, dpa, dpb = arr(dark)
    gap_a, gap_b = _ld_gap(ly, lpa, dy, dpa), _ld_gap(ly, lpb, dy, dpb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        li = rng.integers(0, len(ly), len(ly))
        di = rng.integers(0, len(dy), len(dy))
        # Same indices for both models (paired), so only the models differ, not the images.
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

    Tests whether two models have different overall accuracy. Only images where exactly one
    model is right matter. If the models were equally good, those images would split 50/50
    between "only A right" and "only B right"; a small p_value means the split is unlikely
    under that assumption.
    """
    m = df_a[["image_id", "label", "pred"]].merge(df_b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"))
    ca, cb = m["pred_a"] == m["label"], m["pred_b"] == m["label"]  # per-image correctness
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # only A right / only B right
    # Exact binomial test of n01 out of (n01 + n10) against probability 0.5; p=1 if no disagreements.
    p =1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}
