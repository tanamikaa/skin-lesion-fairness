# ---------------------------------------------------------------------------
# hparam_search.py: equal-budget random hyper-parameter search for BOTH models.
#
# Usage:
#   python scripts/hparam_search.py --trials 8 --epochs 10 [--out results/hparam_search.csv]
#   python scripts/hparam_search.py --configs configs/resnet50.yaml configs/vit_b16.yaml --trials 8
#
# What it is for: a hyper-parameter is a setting we choose BEFORE training (learning
# rate, batch size). Random search tries random combinations and keeps the one with
# the best validation score. It is simple and works well when only a few settings matter.
#
# Fair comparison: every model gets the same number of trials, epochs, seed and the same
# sampled (lr multiplier, batch size) pairs. The learning rate (LR) is log-uniform around
# each model's own default LR (x0.1 to x10). The best trial is chosen with validation
# balanced accuracy only (never the test set). Results are appended to the CSV, so an
# interrupted search can resume.
#
# Where it fits in the pipeline: it sits around the TRAIN step. It calls train() many
# times, then you copy the best values into configs/*.yaml and do the final training run.
# Each trial writes to its own folder (results/hparam, run name hp_<model>_t<trial>) so
# it never overwrites the real experiment results.
# ---------------------------------------------------------------------------

# Lets us read options typed on the command line.
import argparse
# Lets us print progress messages.
import logging
# Path makes file and folder paths easy to handle.
from pathlib import Path

# numpy gives random numbers and maths.
import numpy as np
# pandas keeps the results table.
import pandas as pd
# Reads YAML config files.
import yaml

# The function that trains one model.
from skinfair.train import train
# Helper that changes config values using KEY=VALUE text.
from skinfair.utils import apply_overrides


# sample_trials: picks random trial settings.
# Inputs:
#   n: how many trials to make.
#   seed: random seed (int); the same seed always gives the same list.
#   lr_range: (low, high) for the learning-rate multiplier, default 0.1x to 10x.
#   batch_sizes: batch sizes to pick from (a batch = images processed together).
# Returns: a list of dicts {"lr_mult": float, "batch_size": int}.
# Why it exists: the trials are made once and reused for every model, so ResNet and
#   ViT get exactly the same budget and the comparison is fair.
def sample_trials(n, seed, lr_range=(0.1, 10.0), batch_sizes=(16, 32, 64)):
    """
    This function makes random hyper-parameter settings to try.
    Inputs: n trials, seed, lr_range for the learning-rate multiplier, and batch_sizes to pick from.
    It creates a random number generator from the seed so results repeat.
    For each trial it picks a multiplier on a log scale between the two limits.
    It also picks a random batch size from the given choices.
    It returns a list of n dictionaries with lr_mult and batch_size.
    It is needed to search for good training settings fairly and repeatably.
    """
    # Random number generator; the seed makes it repeatable.
    rng = np.random.default_rng(seed)
    # Work in powers of 10, so the multiplier is log-uniform (0.1 to 1 is as likely as 1 to 10).
    lo, hi = np.log10(lr_range[0]), np.log10(lr_range[1])
    # Make n trials, each with a random LR multiplier and a random batch size.
    return [{"lr_mult": float(10 ** rng.uniform(lo, hi)), "batch_size": int(rng.choice(batch_sizes))}
            for _ in range(n)]


# main: runs the whole search.
# Inputs: none directly; the options are read from the command line (see the top of the file).
# Returns: nothing. It writes the results CSV and prints two tables.
# Steps: 1) read options, 2) load the old CSV if any (resume), 3) sample trials once,
#   4) for each model and each trial: build a config, call train(), save one CSV row,
#   5) print all trials and the best trial per model.
# Why it exists: scoring uses validation balanced accuracy only, so the test set stays
#   untouched and the final test result is not tuned on.
def main():
    """
    This function is the entry point of the hyper-parameter search script.
    Input: command-line options such as --configs, --trials, --epochs and --out.
    It makes the same random trials and runs training for each model and trial.
    Finished trials are skipped, and each result is saved to a CSV straight away.
    Each trial records the learning rate, batch size and best validation score.
    It returns nothing; it prints all results and the best setting per model.
    It is needed to choose good settings with an equal budget for every model.
    """
    # Create the object that reads command-line options.
    p = argparse.ArgumentParser()
    # Config files, one per model.
    p.add_argument("--configs", nargs="+", default=["configs/resnet50.yaml", "configs/vit_b16.yaml"])
    # Number of trials per model.
    p.add_argument("--trials", type=int, default=8)
    # Same number of epochs for every trial.
    p.add_argument("--epochs", type=int, default=10, help="equal epoch budget for every trial")
    # Seed used to pick the trials.
    p.add_argument("--seed", type=int, default=42)
    # Where to save the results table.
    p.add_argument("--out", default="results/hparam_search.csv")
    # Any number of KEY=VALUE changes applied to all runs.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra overrides for all runs")
    # Read the options the user typed and store them in args.
    args = p.parse_args()
    # Show INFO messages while running.
    logging.basicConfig(level=logging.INFO)

    # Turn the output path text into a Path object.
    out = Path(args.out)
    # Create the output folder if it does not exist yet.
    out.parent.mkdir(parents=True, exist_ok=True)
    # If the CSV already exists, load it (so we can resume); otherwise start with an empty table.
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(
        columns=["model", "trial", "lr", "batch_size", "epochs", "val_bal_acc", "best_epoch"])
    # Pick the trials once, so every model gets exactly the same ones.
    trials = sample_trials(args.trials, args.seed)

    # Go through each model config file.
    for path in args.configs:
        # Load the base config and apply the --set changes.
        base = apply_overrides(yaml.safe_load(open(path)), args.set)
        # Model name, used in the results table.
        name = base["model"]["name"]
        # k is the trial number, tr is that trial's random settings.
        for k, tr in enumerate(trials):
            # Skip this trial if the CSV already has a result for it with this model.
            if ((done["model"] == name) & (done["trial"] == k)).any():
                # Go to the next trial.
                continue
            # Real LR = this model's default LR times the random multiplier.
            lr = base["train"]["lr"] * tr["lr_mult"]
            # Unique name for this run.
            run = f"hp_{name}_t{k}"
            # Make the trial config: new LR, batch size, epochs, and separate output folders.
            # Patience equals epochs, so there is no early stop and the budget stays equal.
            cfg = apply_overrides(base, [
                f"train.lr={lr}", f"data.batch_size={tr['batch_size']}", f"train.epochs={args.epochs}",
                f"train.early_stopping_patience={args.epochs}",
                "output_dir=results/hparam", f"run_name={run}",
                "checkpoint_dir=results/hparam/checkpoints", "train.progress=false",
            ])
            # Train the model; the summary has best_val_bal_acc and best_epoch, among others.
            summary = train(cfg)
            # Build one result row for the table.
            row = {"model": name, "trial": k, "lr": lr, "batch_size": tr["batch_size"],
                   "epochs": args.epochs, "val_bal_acc": summary["best_val_bal_acc"],
                   "best_epoch": summary["best_epoch"]}
            # Add the new row to the table.
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            # Save after every run, so the search can resume later.
            done.to_csv(out, index=False)

    # Print all trials, grouped by model, best score first.
    print(done.sort_values(["model", "val_bal_acc"], ascending=[True, False]).to_string(index=False))
    # idxmax gives the row of the highest val_bal_acc for each model.
    best =done.loc[done.groupby("model")["val_bal_acc"].idxmax()]
    # Print a header line for the best results.
    print("\nBest per model (copy into configs/*.yaml):")
    # Print the best trial for each model.
    print(best[["model", "lr", "batch_size", "val_bal_acc"]].to_string(index=False))


# Run main() only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Start the search.
    main()
