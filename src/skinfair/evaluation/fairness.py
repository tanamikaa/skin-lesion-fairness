# This file measures fairness across skin tones.
# A model can be good on average but worse for some people, so we score each Fitzpatrick
# skin type (I-VI) separately and compare light skin (I-III) with dark skin (IV-VI).
# It also uses the bootstrap (redrawing the data many times) to check if a gap is real or luck.
# The input is a predictions table with columns label, pred, fitzpatrick (and image_id
# when comparing two models). fitzpatrick = -1 means the skin type is unknown.
# It is called by scripts/evaluate.py and scripts/compare_models.py.

# numpy for arrays, pandas for tables, scipy for the statistics test.
import numpy as np
import pandas as pd
from scipy import stats

# The list of class names used across the project.
from skinfair import CLASSES

# Metric helpers from our own metrics file.
from .metrics import classification_metrics, fast_balanced_accuracy, per_class_recall

# Skin types I-III count as "light".
LIGHT_FST = (1, 2, 3)
# Skin types IV-VI count as "dark".
DARK_FST = (4, 5, 6)


# per_group_metrics: computes one metric for each group (e.g. each skin type).
# Inputs: df (table with label, pred and the group column), group_col (which column
#   defines the groups), metric (a key of classification_metrics).
# Returns: a table with columns [group_col, n, metric], one row per group, sorted by group.
# Why: shows how the model does on each group; small n means less trustworthy.
def per_group_metrics(df: pd.DataFrame, group_col="fitzpatrick", metric="balanced_accuracy"):
    """
    Computes one metric separately for each group, such as each skin type.
    Steps: drop unknown skin types, split the table by group, score each group.
    Then it builds a table and sorts it by group.
    Inputs: df (label, pred, group column), group_col and metric name.
    Output: a table with columns group, n and the metric, one row per group.
    It shows how the model does for each group; a small n means less trust.
    """
    # For skin type, remove rows where the type is unknown (-1).
    if group_col == "fitzpatrick":
        # Keep only rows with a known skin type.
        df = df[df[group_col] >= 1]
    # Build one row per group: its name, its size, and the metric on its rows only.
    # df.groupby splits the table into one small table (sub) per group value g.
    rows = [
        {group_col: g, "n": len(sub), metric: classification_metrics(sub["label"], sub["pred"])[metric]}
        for g, sub in df.groupby(group_col)
    ]
    # Make a table, sort it by group and tidy the row numbers.
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


# fairness_gap: gap (best minus worst) and ratio (worst / best) across all groups.
# Input: group_df (table from per_group_metrics), metric (column to compare).
# Returns: {"gap": best - worst, "ratio": worst / best}. Example: 0.80, 0.70, 0.60 -> gap 0.20, ratio 0.75.
# Why: one simple number showing how unequal the groups are. It is always >= 0 and does
#   not say which group is worse (the light-minus-dark gap is in bootstrap_gap_ci).
def fairness_gap(group_df: pd.DataFrame, metric="balanced_accuracy"):
    """
    Measures how unequal the groups are with two simple numbers.
    Steps: take the metric values of all groups, find the best and the worst.
    The gap is best minus worst; the ratio is worst divided by best.
    Inputs: group_df (from per_group_metrics) and the metric column name.
    Output: a dict with gap and ratio; the ratio is NaN if the best value is 0.
    It gives one easy number for fairness, but does not say which group is worse.
    """
    # Get the metric values of all groups as an array.
    vals = group_df[metric].to_numpy()
    # Gap: 0 means equal. Ratio: 1.0 means equal; NaN if the best value is 0.
    return {"gap": float(vals.max() - vals.min()), "ratio": float(vals.min() / vals.max()) if vals.max() > 0 else float("nan")}


# light_vs_dark: like per_group_metrics but with only two groups, light and dark.
# Inputs: df (predictions table), metric (which metric to compute).
# Returns: a table with rows "dark" and "light" and columns tone, n, metric.
# Why: skin types IV-VI have few images, so merging them gives a more stable number.
def light_vs_dark(df: pd.DataFrame, metric="balanced_accuracy"):
    """
    Computes a metric for just two groups: light skin and dark skin.
    Steps: drop unknown skin types, label types 1-3 light and 4-6 dark.
    Then it reuses per_group_metrics on the new tone column.
    Inputs: df (predictions table) and the metric name.
    Output: a table with rows dark and light and columns tone, n and the metric.
    Merging types IV-VI gives a more stable number because they have few images.
    """
    # Drop unknown skin types; copy so adding a column is safe.
    d = df[df["fitzpatrick"] >= 1].copy()
    # Mark each row as light (types 1-3) or dark (types 4-6).
    d["tone"] = np.where(d["fitzpatrick"] <= 3, "light", "dark")
    # Reuse the per-group function on the new "tone" column.
    return per_group_metrics(d, group_col="tone", metric=metric)


# per_class_recall_light_vs_dark: recall of each disease class for light and dark skin.
# Recall = of the images that truly are a class, the share the model found.
# Input: df (table with label, pred, fitzpatrick).
# Returns: a table with columns class, n_light, n_dark, recall_light, recall_dark, gap
#   (gap = light minus dark; positive = worse on dark). Recall is NaN if a group has no
#   image of that class, and then the gap is NaN too.
# Why: shows WHICH disease classes the model misses more on dark skin.
def per_class_recall_light_vs_dark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the recall of each disease class for light and for dark skin.
    Steps: split into light and dark rows, compute recall per class for each group.
    It also counts the images per class and subtracts dark recall from light recall.
    Input: df with label, pred and fitzpatrick columns.
    Output: a table with class, n_light, n_dark, recall_light, recall_dark and gap.
    It shows which disease classes the model misses more on dark skin.
    """
    # Split the data into light and dark rows.
    light, dark = _split_light_dark(df)
    # Recall for each class in the light group.
    rl = per_class_recall(light["label"], light["pred"])
    # Recall for each class in the dark group.
    rd_ = per_class_recall(dark["label"], dark["pred"])
    # Number of images per class in the light group.
    nl = np.bincount(light["label"], minlength=len(CLASSES))
    # Number of images per class in the dark group.
    nd = np.bincount(dark["label"], minlength=len(CLASSES))
    # Put everything in one table, one row per class.
    out = pd.DataFrame({"class": CLASSES, "n_light": nl, "n_dark": nd, "recall_light": rl, "recall_dark": rd_})
    # Gap between the two groups for each class (NaN - number = NaN, so absent classes stay NaN).
    out["gap"] = out["recall_light"] - out["recall_dark"]
    # Give back the finished table.
    return out


# _split_light_dark: splits rows into light and dark groups.
# Input: df (table with a fitzpatrick column).
# Returns: two tables (light rows, dark rows); unknown skin types (-1) are dropped.
# Why: a gap needs both groups, so it raises an error if either one is empty.
def _split_light_dark(df):
    """
    Splits the rows into a light-skin group and a dark-skin group.
    Steps: drop unknown skin types, select types 1-3 as light and 4-6 as dark.
    It raises an error if either group is empty.
    Input: df with a fitzpatrick column.
    Output: two tables, the light rows and the dark rows.
    A gap needs both groups, so this check is needed.
    """
    # Remove rows with unknown skin type.
    d =df[df["fitzpatrick"] >= 1]
    # Rows with light skin types.
    light = d[d["fitzpatrick"].isin(LIGHT_FST)]
    # Rows with dark skin types.
    dark = d[d["fitzpatrick"].isin(DARK_FST)]
    # We cannot compare groups if one of them has no rows.
    if len(light) == 0 or len(dark) == 0:
        # Stop with a clear message.
        raise ValueError("Need rows in both light (I-III) and dark (IV-VI) groups")
    # Return both groups.
    return light, dark


# _ld_gap: light minus dark balanced accuracy.
# Inputs: true labels and predictions (1-D arrays) for the light group, then the dark group.
# Returns: one float; positive means worse on dark skin. Example: 0.80 - 0.65 = +0.15.
# Why: this is the basic gap number reused by the bootstrap functions.
def _ld_gap(light_y, light_p, dark_y, dark_p):
    """
    Computes the light minus dark balanced accuracy.
    Steps: score the light group, score the dark group, then subtract.
    Inputs: true labels and predictions for the light group, then for the dark group.
    The inputs are 1-D arrays.
    Output: one float; positive means the model is worse on dark skin.
    It is the basic gap number reused by the bootstrap functions.
    """
    # Score each group the same way, then subtract.
    return fast_balanced_accuracy(light_y, light_p) - fast_balanced_accuracy(dark_y, dark_p)


# bootstrap_gap_ci: confidence interval (CI) of the light-vs-dark gap using the bootstrap.
# The bootstrap redraws the data many times to see how much the gap changes by luck.
# Inputs: df (label, pred, fitzpatrick), n_boot (number of redraws), metric (only
#   "balanced_accuracy" works), seed (makes results repeatable), alpha (0.05 = 95% CI).
# Returns: dict with gap (real data), ci_low, ci_high, n_light, n_dark, n_boot.
# Why: tells us if "worse on dark skin" is a real finding; if the CI excludes 0 it is
#   unlikely to be luck. Light and dark are redrawn separately so each keeps its size.
def bootstrap_gap_ci(df, n_boot=1000, metric="balanced_accuracy", seed=0, alpha=0.05):
    """
    Finds a confidence interval for the light-vs-dark gap using the bootstrap.
    Steps: split the data, then redraw each group many times with replacement.
    Each redraw gives a gap; the middle percentiles of all gaps form the interval.
    Inputs: df, n_boot, metric (only balanced accuracy), seed and alpha.
    Output: dict with gap, ci_low, ci_high, n_light, n_dark and n_boot.
    It tells us if the gap is a real finding or just luck.
    """
    # Only balanced accuracy works with the fast code used here.
    if metric != "balanced_accuracy":
        # Stop with a clear message.
        raise NotImplementedError("bootstrap_gap_ci supports metric='balanced_accuracy' only")
    # Split the data into light and dark rows.
    light, dark = _split_light_dark(df)
    # Light group: true labels and predictions as arrays.
    ly, lp = light["label"].to_numpy(), light["pred"].to_numpy()
    # Dark group: true labels and predictions as arrays.
    dy, dp = dark["label"].to_numpy(), dark["pred"].to_numpy()
    # Seeded random generator so results can be repeated.
    rng = np.random.default_rng(seed)
    # Empty array to store the gap from each resample.
    gaps = np.empty(n_boot)
    # Repeat the resampling n_boot times.
    for b in range(n_boot):
        # Pick random rows with replacement, same size as the light group.
        # (indices like [3, 3, 0, 7]: row 3 may appear twice, row 1 not at all)
        li = rng.integers(0, len(ly), len(ly))
        # Same for the dark group.
        di = rng.integers(0, len(dy), len(dy))
        # Gap computed on the resampled rows.
        gaps[b] = _ld_gap(ly[li], lp[li], dy[di], dp[di])
    # Take the low and high percentiles (2.5 and 97.5 when alpha is 0.05).
    # Cutting 2.5% from each end leaves the middle 95% of the resampled gaps.
    lo, hi = np.percentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Return the results as a dict; each entry is explained below.
    return {
        # Gap on the real data (not resampled).
        "gap": float(_ld_gap(ly, lp, dy, dp)),
        # Lower end of the CI.
        "ci_low": float(lo),
        # Upper end of the CI.
        "ci_high": float(hi),
        # Number of light images, so readers can judge how much to trust the result.
        "n_light": int(len(ly)),
        # Number of dark images.
        "n_dark": int(len(dy)),
        # How many redraws were used.
        "n_boot": n_boot,
    }


# bootstrap_group_ci: balanced accuracy per group with a bootstrap CI.
# Inputs: df (predictions table), group_col (group column), n_boot (redraws), seed, alpha.
# Returns: a table with columns [group_col, n, value, ci_low, ci_high], one row per group;
#   value is the balanced accuracy on the real data.
# Why: used for the error bars in plots; a wide CI means the group is too small to trust.
def bootstrap_group_ci(df, group_col="fitzpatrick", n_boot=1000, seed=0, alpha=0.05):
    """
    Computes balanced accuracy per group with a bootstrap confidence interval.
    Steps: drop unknown skin types, then handle each group on its own.
    For each group it redraws the rows many times and scores every redraw.
    Inputs: df, group_col, n_boot, seed and alpha.
    Output: a table with group, n, value, ci_low and ci_high, one row per group.
    It gives the error bars for plots; a wide interval means a small, uncertain group.
    """
    # For skin type, remove rows where the type is unknown (-1).
    if group_col == "fitzpatrick":
        # Keep only rows with a known skin type.
        df = df[df[group_col] >= 1]
    # Seeded random generator so results can be repeated.
    rng = np.random.default_rng(seed)
    # This list will hold one result row per group.
    rows = []
    # Handle each group on its own.
    for g, sub in df.groupby(group_col):
        # True labels and predictions of this group as arrays.
        y, p = sub["label"].to_numpy(), sub["pred"].to_numpy()
        # Resample the group n_boot times and score each resample.
        # The inner generator makes one random index list per redraw; y[i], p[i] pick those rows.
        boots = np.array([
            fast_balanced_accuracy(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
        ])
        # Low and high percentiles of the resampled scores.
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        # Save group name, size, real score and the CI.
        rows.append({group_col: g, "n": len(y), "value": fast_balanced_accuracy(y, p),
                     "ci_low": float(lo), "ci_high": float(hi)})
    # Make a sorted table from the rows.
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


# compare_gap_bootstrap: tests if model A's light-vs-dark gap differs from model B's.
# The test is paired: both models are scored on the same images, matched by image_id.
# Inputs: df_a, df_b (predictions tables; both need image_id, df_a also gives label and
#   fitzpatrick), n_boot, seed, alpha (as in bootstrap_gap_ci).
# Returns: dict with gap_a, gap_b, diff (= gap_a - gap_b), ci_low, ci_high, p_value,
#   n_matched (images both models predicted). diff < 0 means A is fairer.
# Why: same resampled images for both models mean the difference reflects the models,
#   not which images were drawn. A small p_value (below 0.05) suggests a real difference.
def compare_gap_bootstrap(df_a, df_b, n_boot=2000, seed=0, alpha=0.05):
    """
    Tests whether model A and model B have different light-vs-dark gaps.
    Steps: match the two tables by image_id, split light and dark, compute both gaps.
    Then it redraws the same images for both models many times and records the difference.
    Inputs: df_a, df_b, n_boot, seed and alpha.
    Output: dict with gap_a, gap_b, diff, ci_low, ci_high, p_value and n_matched.
    Using the same images keeps the comparison fair; a small p_value suggests a real difference.
    """
    # Join both tables by image_id so each row has both models' predictions.
    # validate="one_to_one" raises an error if an image_id appears twice.
    m = df_a[["image_id", "label", "pred", "fitzpatrick"]].merge(
        df_b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    # Remove rows with unknown skin type.
    m = m[m["fitzpatrick"] >= 1]
    # Rows with light skin types.
    light = m[m["fitzpatrick"].isin(LIGHT_FST)]
    # Rows with dark skin types.
    dark = m[m["fitzpatrick"].isin(DARK_FST)]
    # We need images in both groups to compare.
    if len(light) == 0 or len(dark) == 0:
        # Stop with a clear message.
        raise ValueError("Need matched rows in both light and dark groups")

    # arr: small helper inside this function.
    # Input: d, a table with label, pred_a and pred_b columns.
    # Returns: true labels, model A predictions and model B predictions as numpy arrays.
    # Why: avoids repeating the same three conversions for light and dark.
    def arr(d):
        """
        A small helper that turns table columns into numpy arrays.
        Steps: take the label, pred_a and pred_b columns and convert each one.
        Input: d, a table with label, pred_a and pred_b columns.
        Output: three numpy arrays: true labels, model A predictions, model B predictions.
        It is called for both the light and the dark group.
        It avoids repeating the same three conversions.
        """
        # Convert the three columns to numpy arrays.
        return d["label"].to_numpy(), d["pred_a"].to_numpy(), d["pred_b"].to_numpy()

    # Arrays for the light group.
    ly, lpa, lpb = arr(light)
    # Arrays for the dark group.
    dy, dpa, dpb = arr(dark)
    # Real gap of model A and model B.
    gap_a, gap_b = _ld_gap(ly, lpa, dy, dpa), _ld_gap(ly, lpb, dy, dpb)
    # Seeded random generator so results can be repeated.
    rng = np.random.default_rng(seed)
    # Empty array to store the gap difference from each resample.
    diffs = np.empty(n_boot)
    # Repeat the resampling n_boot times.
    for b in range(n_boot):
        # Random light rows (with replacement).
        li = rng.integers(0, len(ly), len(ly))
        # Random dark rows (with replacement).
        di = rng.integers(0, len(dy), len(dy))
        # Use the same rows for both models, so only the models differ, not the images.
        diffs[b] = _ld_gap(ly[li], lpa[li], dy[di], dpa[di]) - _ld_gap(ly[li], lpb[li], dy[di], dpb[di])
    # Low and high percentiles of the differences give the CI.
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided p-value: how often the difference falls on one side of 0.
    # If almost all resampled diffs are on one side of 0, the true diff is very unlikely
    # to be 0, so p is small. The factor 2 makes the test two-sided (either model could win).
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    # Return the results as a dict; p-value is capped at 1.0.
    return {
        "gap_a": float(gap_a), "gap_b": float(gap_b), "diff": float(gap_a - gap_b),
        "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0)),
        "n_matched": int(len(m)),
    }


# mcnemar_test: do two models have different accuracy on the same images?
# Only images where exactly one model is right matter; if the models were equally good
# they would split about 50/50. Example: A alone right on 30, B alone on 10 -> small p-value.
# Input: df_a, df_b (predictions tables with image_id, label, pred).
# Returns: dict with a_only_correct, b_only_correct (image counts) and p_value.
# Why: tells us if the accuracy difference between two models is real; it uses an exact
#   binomial test (a fair-coin test) on those counts.
def mcnemar_test(df_a, df_b):
    """
    Tests whether two models differ in accuracy on the same images.
    Steps: match the tables by image_id and check which model is right on each image.
    Count images where only A is right and where only B is right.
    Run an exact binomial test (fair coin) on those two counts.
    Inputs: df_a and df_b with image_id, label and pred.
    Output: dict with a_only_correct, b_only_correct and p_value.
    """
    # Join both tables by image_id.
    m = df_a[["image_id", "label", "pred"]].merge(df_b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"))
    # True/False for each image: was model A right, was model B right.
    ca, cb = m["pred_a"] == m["label"], m["pred_b"] == m["label"]
    # Count images where only A is right, and where only B is right.
    # (images where both are right or both are wrong tell us nothing about who is better)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    # Exact binomial test against 50/50; if the models never disagree, p is 1.
    p =1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    # Return the counts and the p-value.
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}
