"""Usage: python scripts/train.py --config configs/resnet50.yaml"""
import argparse

import yaml

from skinfair.train import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        train(yaml.safe_load(f))
