# This file draws the charts used in the report: scores per skin type, the light-vs-dark
# gap of each model, a confusion matrix, and explanation quality per group.
# It is the last part of the "evaluate" step. It only draws numbers that were already
# computed; it does not compute any metric itself.
# scripts/compare_models.py calls plot_per_fst and plot_gap_comparison.
# Every plot function saves a PNG to `path` and returns that path.
#
# Words used below:
#   - CI (confidence interval): the range that probably holds the true value, drawn as an error bar.
#   - confusion matrix: table with row = true class, column = predicted class, cell = image count.

# Path makes working with file paths easy.
from pathlib import Path

# matplotlib draws the charts.
import matplotlib

# "Agg" draws to files only, so it works on machines with no screen.
# It must be set before pyplot is imported (so the imports below need noqa: E402).
matplotlib.use("Agg")
# pyplot is the main plotting tool.
import matplotlib.pyplot as plt  # noqa: E402
# numpy is used for array maths.
import numpy as np  # noqa: E402

# Class names, used for axis labels.
from skinfair import CLASSES  # noqa: E402


# plot_per_fst: draws grouped bars per skin type, one bar per model, with CI error bars.
# Inputs:
#   group_tables: dict {model name: table from bootstrap_group_ci} with columns
#     fitzpatrick (1-6), value, ci_low, ci_high.
#   path: where to save the PNG.
#   ylabel: text for the y-axis.
# Returns: the saved file path.
# Why it exists: it shows at a glance whether a model scores lower on darker skin types.
def plot_per_fst(group_tables: dict, path, ylabel="Balanced accuracy"):
    """Draw a bar chart of a score for each skin tone (Fitzpatrick type).
    Steps: make a figure, give each model's bars a small offset,
    compute error bars from the confidence interval, draw bars, label axes, then save.
    Inputs: group_tables (dict of model name to table), path to save, y-axis label.
    Returns the saved file path.
    Needed to see visually if some skin tones get worse results.
    """
    # Create an empty figure with one set of axes.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Number of models = number of bars in each skin-type group.
    k = len(group_tables)
    # Width of one bar; all k bars together fill 0.8 of the space.
    width = 0.8 / k
    # Draw the bars for each model in turn.
    for j, (name, t) in enumerate(group_tables.items()):
        # Shift each model's bars left or right so they sit side by side.
        x = t["fitzpatrick"].to_numpy() + (j - (k - 1) / 2) * width
        # Error bar sizes: distance below and above the bar top.
        err = np.vstack([t["value"] - t["ci_low"], t["ci_high"] - t["value"]])
        # Draw this model's bars with error bars and a legend label.
        ax.bar(x, t["value"], width, yerr=err, capsize=3, label=name)
    # Label the x-axis.
    ax.set_xlabel("Fitzpatrick skin type")
    # Label the y-axis.
    ax.set_ylabel(ylabel)
    # Put one tick for each skin type 1 to 6.
    ax.set_xticks(range(1, 7))
    # Show skin types as Roman numerals.
    ax.set_xticklabels(["I", "II", "III", "IV", "V", "VI"])
    # Show which colour is which model.
    ax.legend()
    # Add a title.
    ax.set_title("Performance per skin type (95% bootstrap CI)")
    # Save the figure and return its path.
    return _save(fig, path)


# plot_gap_comparison: draws one bar per model showing its light-minus-dark gap.
# Above zero means the model is worse on dark skin.
# Inputs:
#   gaps: dict {model name: dict from bootstrap_gap_ci} with keys gap, ci_low, ci_high.
#   path: where to save the PNG.
# Returns: the saved file path.
# Why it exists: if an error bar crosses zero, the gap may be just luck, and this shows it.
def plot_gap_comparison(gaps: dict, path):
    """Draw a bar chart comparing the fairness gap of different models.
    Steps: read each model's gap and confidence interval from the dict,
    turn them into error bars, draw coloured bars with a zero line, then save.
    Inputs: gaps (dict of name to gap, ci_low, ci_high) and the save path.
    Returns the saved file path.
    Needed to compare easily which model is fairer.
    """
    # Create a small figure.
    fig, ax = plt.subplots(figsize=(4.5, 4))
    # Model names are the dict keys.
    names = list(gaps)
    # Bar heights: the gap of each model.
    vals = np.array([gaps[n]["gap"] for n in names])
    # Error bar sizes: distance down to ci_low and up to ci_high, shape [2, n_models].
    err =np.array([[gaps[n]["gap"] - gaps[n]["ci_low"], gaps[n]["ci_high"] - gaps[n]["gap"]] for n in names]).T
    # Draw the bars with error bars, one colour per model.
    ax.bar(names, vals, yerr=err, capsize=4, color=["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(names)])
    # Draw a black line at zero, which means no gap.
    ax.axhline(0, color="k", lw=0.8)
    # Label the y-axis.
    ax.set_ylabel("Light − dark balanced accuracy")
    # Add the title.
    ax.set_title("Skin-tone gap (positive = worse on dark)")
    # Save the figure and return its path.
    return _save(fig, path)


# plot_confusion: draws a heat-map of a confusion matrix.
# Colour shows the share of each true class; the text in each cell shows the raw count.
# Inputs:
#   cm: 7x7 list or array of counts (row = true class, column = predicted class).
#   path: where to save the PNG.
#   title: chart title.
# Returns: the saved file path.
# Why colour uses shares: a big class (like nv) would otherwise look dark everywhere
# and hide how well the small classes are recognised.
def plot_confusion(cm, path, title="Confusion matrix"):
    """Draw a confusion matrix as a coloured heatmap.
    Steps: normalise each row so it sums to 1, show it as an image,
    write the class names on both axes and the values in each cell, then save.
    Inputs: cm (7 x 7 counts), the save path and a title.
    Returns the saved file path.
    Needed to see which classes get mixed up with each other.
    """
    # Make sure the matrix is a float array.
    cm = np.asarray(cm, dtype=float)
    # Divide each row by its total (at least 1 to avoid 0/0), so each row sums to 1.
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    # Create the figure.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Draw the normalised matrix as coloured squares.
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    # Label the columns with class names, turned 45 degrees.
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    # Label the rows with class names.
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    # Label the x-axis.
    ax.set_xlabel("Predicted")
    # Label the y-axis.
    ax.set_ylabel("True")
    # Visit every row i.
    for i in range(len(CLASSES)):
        # Visit every column j.
        for j in range(len(CLASSES)):
            # Write the count in the cell; white text on dark cells so it is easy to read.
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    # Add a colour scale.
    fig.colorbar(im, ax=ax, fraction=0.046)
    # Add the title.
    ax.set_title(title)
    # Save the figure and return its path.
    return _save(fig, path)


# plot_quality_by_group: draws grouped bars, one group per skin type, one bar per explanation method.
# Inputs:
#   table: pandas table with columns method, group, value (one row per method and group).
#   path: where to save the PNG.
#   ylabel: y-axis text (default "AUC", the deletion/insertion score from explain/quality.py;
#     for insertion higher is better, for deletion lower is better).
# Returns: the saved file path.
def plot_quality_by_group(table, path, ylabel="AUC"):
    """Draw grouped bars of an explanation-quality score for each group.
    Steps: list the methods and groups, then for each method pick its rows,
    draw bars with error bars at offset positions, label axes and legend, then save.
    Inputs: table (method, group, value, sem), the save path and a y-axis label.
    Returns the saved file path.
    Needed to check if explanations are equally good for every skin tone.
    """
    # List of methods (in order of appearance).
    methods = list(table["method"].unique())
    # Sorted list of groups.
    groups = sorted(table["group"].unique())
    # Create the figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Width of one bar so all methods fit in 0.8 of a group.
    width = 0.8 / len(methods)
    # Draw the bars for each method in turn.
    for j, m in enumerate(methods):
        # This method's rows in the same order as `groups` (a missing group gives no bar).
        sub = table[table["method"] == m].set_index("group").reindex(groups)
        # Draw this method's bars, shifted so the methods sit side by side.
        ax.bar(np.arange(len(groups)) + (j - (len(methods) - 1) / 2) * width, sub["value"], width, label=m)
    # One tick per group, labelled with the group name.
    ax.set_xticks(range(len(groups)), [str(g) for g in groups])
    # Label the y-axis.
    ax.set_ylabel(ylabel)
    # Show which colour is which method.
    ax.legend()
    # Save the figure and return its path.
    return _save(fig, path)


# _save: saves a figure to disk, closes it, and returns the path.
# Inputs:
#   fig: the matplotlib figure to save.
#   path: where to save it (folders are created if missing).
# Returns: the path as a Path object.
# Why it exists: closing figures matters because matplotlib keeps open ones in memory.
def _save(fig, path):
    """Finish a figure and write it to disk.
    Steps: make the parent folder if missing, tighten the layout,
    save the image at 150 dpi, and close the figure to free memory.
    Inputs: fig (a matplotlib figure) and path (where to save it).
    Returns the path as a Path object.
    Needed so all plot functions save their files in the same way.
    """
    # Make sure we have a Path object.
    path = Path(path)
    # Create the output folder if it is missing.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Adjust spacing so labels are not cut off.
    fig.tight_layout()
    # Write the image file.
    fig.savefig(path, dpi=150)
    # Close the figure to free memory.
    plt.close(fig)
    # Give back the saved path.
    return path
