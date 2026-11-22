"""Figures for the report. Owner: Karanpal. All functions save to `path` and return it."""
from pathlib import Path

import matplotlib

# "Agg" backend draws to files only (no window), so plotting works on servers without a screen.
# It must be set before pyplot is imported, hence the noqa: E402 on the imports below.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skinfair import CLASSES  # noqa: E402


def plot_per_fst(group_tables: dict, path, ylabel="Balanced accuracy"):
    """group_tables: {model_name: DataFrame[fitzpatrick, n, value, ci_low, ci_high]} -> grouped bars with CI.

    Draws one bar per (skin type, model) with error bars for the confidence interval, so
    models can be compared per skin type. Tables usually come from bootstrap_group_ci.
    Saves the figure to `path` and returns the path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    k = len(group_tables)  # number of models = bars per skin type
    width = 0.8 / k  # all k bars together fill 0.8 of the space for one skin type
    for j, (name, t) in enumerate(group_tables.items()):
        # Shift model j's bars left/right so the k bars sit side by side, centred on the FST.
        x = t["fitzpatrick"].to_numpy() + (j - (k - 1) / 2) * width
        # matplotlib wants error sizes as [distance below, distance above] the bar top.
        err = np.vstack([t["value"] - t["ci_low"], t["ci_high"] - t["value"]])
        ax.bar(x, t["value"], width, yerr=err, capsize=3, label=name)
    ax.set_xlabel("Fitzpatrick skin type")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["I", "II", "III", "IV", "V", "VI"])
    ax.legend()
    ax.set_title("Performance per skin type (95% bootstrap CI)")
    return _save(fig, path)


def plot_gap_comparison(gaps: dict, path):
    """gaps: {model_name: dict(gap, ci_low, ci_high)} from bootstrap_gap_ci.

    One bar per model showing its light - dark gap with CI error bars. Bars above the
    zero line mean the model is worse on dark skin. Saves to `path` and returns it.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4))
    names = list(gaps)  # model names (dict keys)
    vals = np.array([gaps[n]["gap"] for n in names])
    # Shape [2, n_models]: row 0 = distance down to ci_low, row 1 = distance up to ci_high.
    err =np.array([[gaps[n]["gap"] - gaps[n]["ci_low"], gaps[n]["ci_high"] - gaps[n]["gap"]] for n in names]).T
    ax.bar(names, vals, yerr=err, capsize=4, color=["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(names)])
    ax.axhline(0, color="k", lw=0.8)  # zero line = no gap
    ax.set_ylabel("Light − dark balanced accuracy")
    ax.set_title("Skin-tone gap (positive = worse on dark)")
    return _save(fig, path)


def plot_confusion(cm, path, title="Confusion matrix"):
    """Draw a confusion matrix (rows = true class, columns = predicted class) as a heat-map.

    Colours show the row-normalised value (fraction of each true class), the text in each
    cell shows the raw count. Saves to `path` and returns it.
    """
    cm = np.asarray(cm, dtype=float)
    # Divide each row by its total (min 1 to avoid 0/0) -> each row sums to 1 = per-class recall.
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            # Write the count in the cell; white text on dark cells so it stays readable.
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    return _save(fig, path)


def plot_quality_by_group(table, path, ylabel="AUC"):
    """table: DataFrame with columns method, group, value (e.g. deletion AUC per FST per method).

    Grouped bar chart: one group of bars per skin type, one bar per explanation method.
    Shows whether explanation quality differs between skin types. Saves to `path`, returns it.
    """
    methods = list(table["method"].unique())
    groups = sorted(table["group"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        # Rows of this method, ordered like `groups` (missing groups become NaN = no bar).
        sub = table[table["method"] == m].set_index("group").reindex(groups)
        ax.bar(np.arange(len(groups)) + (j - (len(methods) - 1) / 2) * width, sub["value"], width, label=m)
    ax.set_xticks(range(len(groups)), [str(g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.legend()
    return _save(fig, path)


def _save(fig, path):
    """Save `fig` to `path` (creating folders if needed), close it, and return the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create output folder if missing
    fig.tight_layout()  # adjust spacing so labels are not cut off
    fig.savefig(path, dpi=150)
    plt.close(fig)  # free memory; important when making many figures
    return path
