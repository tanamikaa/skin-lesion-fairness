# ---------------------------------------------------------------------------
# train.py: command-line entry point for training ONE model.
#
# Usage:
#   python scripts/train.py --config configs/resnet50.yaml [--set train.lr=3e-4 loss.name=ce ...]
#
# What it does: reads a YAML config file, applies any --set changes on top of it,
# and calls train() from src/skinfair/train.py, which does the real work.
# `--set` lets you change any config value (using a dotted path like train.lr)
# without copying the config file.
#
# Where it fits in the pipeline: data -> model -> TRAIN -> evaluate -> explain.
# Its output (best checkpoint + log.csv) is used by scripts/evaluate.py and
# scripts/explain.py.
# ---------------------------------------------------------------------------

# Lets us read options typed on the command line.
import argparse
# Lets us print progress messages while training runs.
import logging

# Reads YAML config files into Python dicts.
import yaml

# The function that trains the model.
from skinfair.train import train
# Helper that changes config values using KEY=VALUE text from the command line.
from skinfair.utils import apply_overrides

# Run the code below only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Show progress messages with the time in front of each one.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Create the object that reads command-line options.
    p = argparse.ArgumentParser()
    # Path to the YAML file with all settings (the user must give it).
    p.add_argument("--config", required=True)
    # Any number of KEY=VALUE changes (for example loss.name=ce); an empty list if none are given.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    # Read the options the user typed and store them in args.
    args = p.parse_args()
    # Open the config file for reading.
    with open(args.config) as f:
        # Load the YAML into a dict, then apply the --set changes to it
        # (apply_overrides works on a copy, so the file itself is never changed).
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    # Train the model and print the summary dict that train() returns.
    print(train(cfg))
