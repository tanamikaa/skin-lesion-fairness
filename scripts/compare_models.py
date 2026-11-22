# ---------------------------------------------------------------------------
# compare_models.py: compares the fairness of two models (e.g. CNN vs ViT).
#
# Usage:
#   python scripts/compare_models.py \
#       --a results/resnet50/predictions_fitz.csv --name-a ResNet-50 \
#       --b results/vit_b16/predictions_fitz.csv  --name-b ViT-B/16 --out reports/figures
#
# What it does: it answers "is model A fairer across skin tones than model B?".
# It reads the prediction CSVs that evaluate.py already wrote, so no model is run here.
# It writes (in --out): fst_comparison.png, gap_comparison.png and
# fairness_comparison.json (gaps with bootstrap confidence intervals, a paired
# bootstrap test on the gap difference, and a McNemar test).
#
# Where it fits in the pipeline (data -> model -> train -> evaluate -> explain):
# it is the last "evaluate" step, run after evaluate.py has run for both models.
#
# How to read the JSON:
#   - gap = light balanced accuracy - dark balanced accuracy (positive = worse on dark skin).
#   - diff = gap_a - gap_b; negative means A has the smaller gap (A is fairer).
#   - p_value below 0.05 suggests the difference is real and not luck.
#   - McNemar compares the images where only one of the two models is correct.
#   - Both CSVs must cover the SAME images (matched by image_id) for the paired tests.
# ---------------------------------------------------------------------------

# Lets us read options typed on the command line.
import argparse
# Used to turn the results into JSON text.
import json
# Path makes file and folder paths easy to handle.
from pathlib import Path

# pandas reads the CSV tables.
import pandas as pd

# Fairness helpers: confidence intervals, model comparison, and the McNemar test.
from skinfair.evaluation.fairness import (
    bootstrap_gap_ci,
    bootstrap_group_ci,
    compare_gap_bootstrap,
    mcnemar_test,
)
# Plot helpers that draw the two comparison charts.
from skinfair.evaluation.plots import plot_gap_comparison, plot_per_fst

# Run the code below only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Create the object that reads command-line options.
    p = argparse.ArgumentParser()
    # CSV of predictions for model A (must be given).
    p.add_argument("--a", required=True)
    # CSV of predictions for model B (must be given).
    p.add_argument("--b", required=True)
    # Display name of model A in the charts.
    p.add_argument("--name-a", default="A")
    # Display name of model B in the charts.
    p.add_argument("--name-b", default="B")
    # Folder where the outputs are saved.
    p.add_argument("--out", default="reports/figures")
    # How many bootstrap resamples to use (more = slower but steadier numbers).
    p.add_argument("--n-boot", type=int, default=2000)
    # Read the options the user typed and store them in args.
    args = p.parse_args()

    # Load both prediction tables (columns: image_id, label, pred, fitzpatrick).
    a, b = pd.read_csv(args.a), pd.read_csv(args.b)
    # Turn the output folder text into a Path object.
    out = Path(args.out)
    # Create the output folder if it does not exist yet.
    out.mkdir(parents=True, exist_ok=True)

    # For each model, compute balanced accuracy per skin type with confidence intervals.
    tables = {args.name_a: bootstrap_group_ci(a, n_boot=args.n_boot),
              args.name_b: bootstrap_group_ci(b, n_boot=args.n_boot)}
    # For each model, compute the light-minus-dark accuracy gap with a confidence interval.
    gaps = {args.name_a: bootstrap_gap_ci(a, n_boot=args.n_boot),
            args.name_b: bootstrap_gap_ci(b, n_boot=args.n_boot)}
    # Draw and save the per-skin-type chart.
    plot_per_fst(tables, out / "fst_comparison.png")
    # Draw and save the gap chart.
    plot_gap_comparison(gaps, out / "gap_comparison.png")

    # Collect everything we want to save in one dict.
    # "gaps" holds each model's own gap.
    # "paired_gap_difference" tests whether the two gaps really differ (not just by chance);
    #   "paired" means both models are scored on the same resampled images.
    # "mcnemar_accuracy" tests whether the models are right on different images
    #   (exact McNemar test, using the images where exactly one model is correct).
    # "note" is a short help text so readers know how to read the numbers.
    result = {
        "gaps": gaps,
        "paired_gap_difference": compare_gap_bootstrap(a, b, n_boot=args.n_boot),
        "mcnemar_accuracy": mcnemar_test(a, b),
        "note": "diff = gap_a - gap_b; negative means model A is fairer. Gap = light(I-III) - dark(IV-VI) balanced accuracy.",
    }
    # Save the results as a JSON file with nice indentation.
    (out / "fairness_comparison.json").write_text(json.dumps(result, indent=2))
    # Also print the same JSON on screen.
    print(json.dumps(result, indent=2))
