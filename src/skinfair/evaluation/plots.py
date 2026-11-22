"""Figures for the report. Owner: Karanpal. All functions save to `path` and return it."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skinfair import CLASSES  # noqa: E402


def plot_per_fst(group_tables: dict, path, ylabel="Balanced accuracy"):
    """group_tables: {model_name: DataFrame[fitzpatrick, n, value, ci_low, ci_high]} -> grouped bars with CI."""
    fig, ax = plt.subplots(figsize=(7, 4))
    k = len(group_tables)
    width = 0.8 / k
    for j, (name, t) in enumerate(group_tables.items()):
        x = t["fitzpatrick"].to_numpy() + (j - (k - 1) / 2) * width
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
    """gaps: {model_name: dict(gap, ci_low, ci_high)} from bootstrap_gap_ci."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    names = list(gaps)
    vals = np.array([gaps[n]["gap"] for n in names])
    err = np.array([[gaps[n]["gap"] - gaps[n]["ci_low"], gaps[n]["ci_high"] - gaps[n]["gap"]] for n in names]).T
    ax.bar(names, vals, yerr=err, capsize=4, color=["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(names)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Light − dark balanced accuracy")
    ax.set_title("Skin-tone gap (positive = worse on dark)")
    return _save(fig, path)


def plot_confusion(cm, path, title="Confusion matrix"):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    return _save(fig, path)


def plot_quality_by_group(table, path, ylabel="AUC"):
    """table: DataFrame with columns method, group, value (e.g. deletion AUC per FST per method)."""
    methods = list(table["method"].unique())
    groups = sorted(table["group"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        sub = table[table["method"] == m].set_index("group").reindex(groups)
        ax.bar(np.arange(len(groups)) + (j - (len(methods) - 1) / 2) * width, sub["value"], width, label=m)
    ax.set_xticks(range(len(groups)), [str(g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.legend()
    return _save(fig, path)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
