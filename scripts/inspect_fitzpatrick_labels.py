"""List unique Fitzpatrick17k `label` values not yet mapped to a HAM10000 class.

Usage: python scripts/inspect_fitzpatrick_labels.py --csv data/fitzpatrick17k/fitzpatrick17k.csv

Owner: Afzal Raza -- use this to finish FITZ_TO_HAM in
src/skinfair/data/fitzpatrick_mapping.py (currently only "nv" and "vasc"
are partially filled in).
"""
import argparse

import pandas as pd

from skinfair.data.fitzpatrick_mapping import FITZ_TO_HAM

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    counts = df["label"].str.lower().value_counts()
    mapped = set(FITZ_TO_HAM)

    print(f"{len(counts)} unique labels, {len(mapped)} currently mapped.\n")
    print("Unmapped labels (label: count):")
    for label, n in counts.items():
        if label not in mapped:
            print(f"  {label!r}: {n}")
