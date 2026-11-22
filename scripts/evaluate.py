"""Usage: python scripts/evaluate.py --config ... --checkpoint ...    Owner: Karanpal

Loads the model, predicts on HAM10000 test + Fitzpatrick17k, and writes to <output_dir>/<run_name>/:
  predictions_ham_test.csv, predictions_fitz.csv   (image_id, label, pred, prob_0..prob_6, fitzpatrick)
  metrics.json  (overall + per Fitzpatrick type + gap/ratio + light-vs-dark + bootstrap CIs)

Fitzpatrick17k is a *different domain* (clinical photos, subset of classes): read its numbers as
a skin-tone robustness probe, not as test-set performance.

Arguments:
  --config      YAML config file (required)
  --checkpoint  trained model weights file (.pt) (required)
  --n-boot      number of bootstrap resamples for confidence intervals (default 1000)
  --set         optional KEY=VALUE config overrides
"""
import argparse
import json
import logging
import math
from pathlib import Path

import torch
import yaml

from skinfair.data.datasets import get_dataloaders
from skinfair.evaluation.fairness import (
    bootstrap_gap_ci,
    bootstrap_group_ci,
    fairness_gap,
    light_vs_dark,
    per_class_recall_light_vs_dark,
    per_group_metrics,
)
from skinfair.evaluation.metrics import classification_metrics
from skinfair.evaluation.predict import predict, probs_of
from skinfair.models import build_model, load_checkpoint
from skinfair.utils import apply_overrides


def _clean(o):
    """Make an object JSON-safe (NaN -> None, numpy scalars -> python).

    Input: any nested mix of dicts, lists/tuples, numbers and numpy values.
    Returns: the same structure with string dict keys, plain Python numbers and None for NaN.
    """
    if isinstance(o, dict):
        # Clean every value recursively; JSON keys must be strings.
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        # numpy / torch scalars have .item() which returns a plain Python number.
        o = o.item()
    # JSON has no NaN, so replace it with None (null in the file).
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def fairness_report(df, n_boot):
    """All fairness numbers for one prediction table with Fitzpatrick labels.

    Inputs: df - prediction table with columns label, pred, fitzpatrick (-1 = unknown);
            n_boot - number of bootstrap resamples for the confidence intervals.
    Returns: a dict with per-skin-type balanced accuracy (with CIs), the max-min gap and ratio,
    light (I-III) vs dark (IV-VI) results, per-class recall light vs dark and a CI for the
    light-minus-dark gap. Parts that cannot be computed hold {"note": <error message>}.
    """
    # Keep only rows whose Fitzpatrick type is known (1..6).
    known = df[df["fitzpatrick"] >= 1]
    if known.empty:
        return {"note": "no rows with a known Fitzpatrick type"}
    # Balanced accuracy for each Fitzpatrick type (unknown rows are ignored inside).
    per_fst = per_group_metrics(df)
    report = {
        # to_dict("records") turns a DataFrame into a list of row dicts (JSON-friendly).
        "n_with_fst": int(len(known)),
        "per_fst": bootstrap_group_ci(df, n_boot=n_boot).to_dict("records"),
        "gap_max_min": fairness_gap(per_fst),
        "light_vs_dark": light_vs_dark(df).to_dict("records"),
    }
    # These two can raise ValueError (e.g. too few light or dark rows); store the message instead.
    try:
        report["per_class_recall_light_vs_dark"] = per_class_recall_light_vs_dark(df).to_dict("records")
    except ValueError as e:
        report["per_class_recall_light_vs_dark"] = {"note": str(e)}
    try:
        report["light_minus_dark_gap_ci"] = bootstrap_gap_ci(df, n_boot=n_boot)
    except ValueError as e:
        report["light_minus_dark_gap_ci"] = {"note": str(e)}
    return report


def main():
    """Evaluate one checkpoint and write predictions + metrics files.

    Reads the command-line arguments (see module docstring), loads the model from the checkpoint,
    predicts on the HAM10000 test set and (if available) Fitzpatrick17k, saves the prediction
    CSVs and metrics.json, and prints the HAM10000 balanced accuracy and macro-F1.
    Returns nothing.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-boot", type=int, default=1000)  # bootstrap resamples for CIs
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    # Use the device from the config if set, otherwise the GPU when available, else the CPU.
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    # Run name: stored in the checkpoint, or else the file name without "_best"
    # (e.g. resnet50_best.pt -> resnet50).
    name = ckpt.get("run_name") or Path(args.checkpoint).stem.removesuffix("_best")
    out_dir = Path(cfg["output_dir"]) / name  # same folder the training run used
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = get_dataloaders(cfg)  # dict of DataLoaders, e.g. "test" and maybe "fitz"

    # Predict on the HAM10000 test split and save one row per image.
    ham = predict(model, loaders["test"], device)
    ham.to_csv(out_dir / "predictions_ham_test.csv", index=False)
    # Overall classification metrics; probs_of() takes the prob_* columns as a probability array.
    metrics = {"checkpoint": args.checkpoint,
               "ham_test": classification_metrics(ham["label"], ham["pred"], probs_of(ham))}

    # The "fitz" loader exists only when the Fitzpatrick17k data was found.
    if "fitz" in loaders:
        fitz = predict(model, loaders["fitz"], device)
        fitz.to_csv(out_dir / "predictions_fitz.csv", index=False)
        metrics["fitzpatrick17k"] = {
            "overall": classification_metrics(fitz["label"], fitz["pred"], probs_of(fitz)),
            # ** merges the fairness report's keys into this dict.
            **fairness_report(fitz, args.n_boot),
        }
    else:
        metrics["fitzpatrick17k"] = {"note": "Fitzpatrick17k not found; skin-tone analysis skipped"}

    # Make everything JSON-safe, then write it as pretty-printed JSON.
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"HAM10000 test: bal-acc={metrics['ham_test']['balanced_accuracy']:.3f} "
          f"macro-F1={metrics['ham_test']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
