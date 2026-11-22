"""Compare CNN vs ViT fairness from their prediction CSVs.    Owner: Karanpal

Usage:
  python scripts/compare_models.py \
      --a results/resnet50/predictions_fitz.csv --name-a ResNet-50 \
      --b results/vit_b16/predictions_fitz.csv  --name-b ViT-B/16 --out reports/figures

Writes (in --out): fst_comparison.png, gap_comparison.png and fairness_comparison.json containing
per-model gaps with bootstrap CIs, the paired bootstrap test on the gap difference, and McNemar.

Arguments:
  --a, --b            prediction CSVs of model A and model B (required)
  --name-a, --name-b  display names for the two models (default "A" and "B")
  --out               output folder (default reports/figures)
  --n-boot            number of bootstrap resamples (default 2000)
The JSON is also printed to the screen.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from skinfair.evaluation.fairness import (
    bootstrap_gap_ci,
    bootstrap_group_ci,
    compare_gap_bootstrap,
    mcnemar_test,
)
from skinfair.evaluation.plots import plot_gap_comparison, plot_per_fst

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--name-a", default="A")
    p.add_argument("--name-b", default="B")
    p.add_argument("--out", default="reports/figures")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    # Load both prediction tables (columns include label, pred, fitzpatrick).
    a, b = pd.read_csv(args.a), pd.read_csv(args.b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Per Fitzpatrick type balanced accuracy with bootstrap CIs, one table per model.
    tables = {args.name_a: bootstrap_group_ci(a, n_boot=args.n_boot),
              args.name_b: bootstrap_group_ci(b, n_boot=args.n_boot)}
    # Light-minus-dark balanced accuracy gap with a bootstrap CI, per model.
    gaps = {args.name_a: bootstrap_gap_ci(a, n_boot=args.n_boot),
            args.name_b: bootstrap_gap_ci(b, n_boot=args.n_boot)}
    plot_per_fst(tables, out / "fst_comparison.png")
    plot_gap_comparison(gaps, out / "gap_comparison.png")

    result = {
        "gaps": gaps,
        # Paired bootstrap: is model A's gap really different from model B's?
        # McNemar: do the two models differ in which images they get right?
        "paired_gap_difference": compare_gap_bootstrap(a, b, n_boot=args.n_boot),
        "mcnemar_accuracy": mcnemar_test(a, b),
        "note": "diff = gap_a - gap_b; negative means model A is fairer. Gap = light(I-III) - dark(IV-VI) balanced accuracy.",
    }
    (out / "fairness_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
