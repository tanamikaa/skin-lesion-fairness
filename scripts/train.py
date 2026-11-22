"""Usage: python scripts/train.py --config configs/resnet50.yaml [--set train.lr=3e-4 loss.name=ce ...]

`--set` overrides any config value (dotted path, YAML-parsed), handy for the CE / weighted CE /
focal and oversampling experiments without copying config files.
"""
import argparse
import logging

import yaml

from skinfair.train import train
from skinfair.utils import apply_overrides

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    print(train(cfg))
