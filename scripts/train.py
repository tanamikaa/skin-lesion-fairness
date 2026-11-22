"""Usage: python scripts/train.py --config configs/resnet50.yaml [--set train.lr=3e-4 loss.name=ce ...]

`--set` overrides any config value (dotted path, YAML-parsed), handy for the CE / weighted CE /
focal and oversampling experiments without copying config files.

What it does: reads a YAML config, applies any `--set` overrides, trains one model with
`skinfair.train.train(cfg)` and prints the summary dict that `train` returns.

Arguments:
  --config  path to a YAML config file (required)
  --set     zero or more KEY=VALUE overrides, e.g. train.lr=3e-4 (optional)
"""
import argparse
import logging

import yaml

from skinfair.train import train
from skinfair.utils import apply_overrides

# Runs only when the file is executed as a script (not when imported).
if __name__ == "__main__":
    # Show INFO log messages with a timestamp in front.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)  # YAML file with all settings
    # nargs="*": accept any number of KEY=VALUE strings; empty list if not given.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    with open(args.config) as f:
        # Load the YAML into a dict, then return a copy with the --set overrides applied.
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    # Train the model and print the summary dict it returns.
    print(train(cfg))
