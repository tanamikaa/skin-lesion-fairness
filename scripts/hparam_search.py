"""Equal-budget random hyper-parameter search for BOTH models.    Owner: Khushi Narwariya

Usage:
  python scripts/hparam_search.py --trials 8 --epochs 10 [--out results/hparam_search.csv]
  python scripts/hparam_search.py --configs configs/resnet50.yaml configs/vit_b16.yaml --trials 8

Fairness of the comparison: every model gets the same number of trials, the same epoch budget,
the same seed, and the *same sampled (lr multiplier, batch size) pairs*. The LR is sampled
log-uniformly around each model's own default LR (x 0.1 .. x 10 in `--lr-range`), because a
ViT and a ResNet have very different sensible LRs. Selection is on validation balanced accuracy
only (never the test set). Results are appended to the CSV so an interrupted search resumes.

Arguments:
  --configs  one or more YAML configs, one per model (default resnet50 + vit_b16)
  --trials   number of trials per model (default 8)
  --epochs   epochs per trial (default 10)
  --seed     random seed for sampling the trials (default 42)
  --out      results CSV path (default results/hparam_search.csv)
  --set      extra KEY=VALUE overrides applied to every run
Note: the `--lr-range` mentioned above is not a command-line flag; the range is the
`lr_range` default (0.1, 10.0) of sample_trials().
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from skinfair.train import train
from skinfair.utils import apply_overrides


def sample_trials(n, seed, lr_range=(0.1, 10.0), batch_sizes=(16, 32, 64)):
    """Randomly sample n hyper-parameter trials.

    Inputs: n - number of trials; seed - random seed (same seed -> same trials);
            lr_range - (min, max) multiplier for the model's default LR;
            batch_sizes - batch sizes to choose from.
    Returns: a list of n dicts {"lr_mult": float, "batch_size": int}.
    """
    rng = np.random.default_rng(seed)
    # Sample the exponent uniformly, so the multiplier is log-uniform (0.1..1 as likely as 1..10).
    lo, hi = np.log10(lr_range[0]), np.log10(lr_range[1])
    return [{"lr_mult": float(10 ** rng.uniform(lo, hi)), "batch_size": int(rng.choice(batch_sizes))}
            for _ in range(n)]


def main():
    """Run the random search for every config and print the results.

    Reads the command-line arguments (see module docstring), trains every (model, trial) pair
    not already in the results CSV, saves the CSV after each run, then prints all results and
    the best trial per model. Returns nothing.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["configs/resnet50.yaml", "configs/vit_b16.yaml"])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10, help="equal epoch budget for every trial")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/hparam_search.csv")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra overrides for all runs")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Load earlier results if the CSV exists (resume), else start with an empty table.
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(
        columns=["model", "trial", "lr", "batch_size", "epochs", "val_bal_acc", "best_epoch"])
    # Sampled once, so every model gets exactly the same trials.
    trials = sample_trials(args.trials, args.seed)

    for path in args.configs:
        base = apply_overrides(yaml.safe_load(open(path)), args.set)
        name = base["model"]["name"]
        for k, tr in enumerate(trials):  # k = trial number
            # Skip trials already in the CSV for this model.
            if ((done["model"] == name) & (done["trial"] == k)).any():
                continue
            # Actual LR = this model's default LR times the sampled multiplier.
            lr = base["train"]["lr"] * tr["lr_mult"]
            run = f"hp_{name}_t{k}"
            # Trial config: new LR / batch size / epochs, and separate output folders.
            cfg = apply_overrides(base, [
                f"train.lr={lr}", f"data.batch_size={tr['batch_size']}", f"train.epochs={args.epochs}",
                f"train.early_stopping_patience={args.epochs}",  # fixed budget, no early stop
                "output_dir=results/hparam", f"run_name={run}",
                "checkpoint_dir=results/hparam/checkpoints", "train.progress=false",
            ])
            summary = train(cfg)  # dict with best_val_bal_acc and best_epoch, among others
            row = {"model": name, "trial": k, "lr": lr, "batch_size": tr["batch_size"],
                   "epochs": args.epochs, "val_bal_acc": summary["best_val_bal_acc"],
                   "best_epoch": summary["best_epoch"]}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out, index=False)  # incremental save => resumable

    # All trials: grouped by model, best validation balanced accuracy first.
    print(done.sort_values(["model", "val_bal_acc"], ascending=[True, False]).to_string(index=False))
    # idxmax gives the row index of the highest val_bal_acc within each model.
    best =done.loc[done.groupby("model")["val_bal_acc"].idxmax()]
    print("\nBest per model (copy into configs/*.yaml):")
    print(best[["model", "lr", "batch_size", "val_bal_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()
