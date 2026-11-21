"""Usage: python scripts/evaluate.py --config ... --checkpoint ...    Owner: Karanpal

Loads the model, predicts on HAM10000 test + Fitzpatrick17k, and writes to cfg["output_dir"]:
  predictions_ham_test.csv, predictions_fitz.csv   (image_id, label, pred, prob_0..prob_6, fitzpatrick)
  metrics.json  (overall + per Fitzpatrick type + gap/ratio + light-vs-dark + bootstrap CIs)

Fitzpatrick17k is a *different domain* (clinical photos, subset of classes): read its numbers as
a skin-tone robustness probe, not as test-set performance.
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
    per_group_metrics,
)
from skinfair.evaluation.metrics import classification_metrics
from skinfair.evaluation.predict import predict, probs_of
from skinfair.models import build_model, load_checkpoint
from skinfair.utils import apply_overrides


def _clean(o):
    """Make an object JSON-safe (NaN -> None, numpy scalars -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def fairness_report(df, n_boot):
    """All fairness numbers for one prediction table with Fitzpatrick labels."""
    known = df[df["fitzpatrick"] >= 1]
    if known.empty:
        return {"note": "no rows with a known Fitzpatrick type"}
    per_fst = per_group_metrics(df)
    report = {
        "n_with_fst": int(len(known)),
        "per_fst": bootstrap_group_ci(df, n_boot=n_boot).to_dict("records"),
        "gap_max_min": fairness_gap(per_fst),
        "light_vs_dark": light_vs_dark(df).to_dict("records"),
    }
    try:
        report["light_minus_dark_gap_ci"] = bootstrap_gap_ci(df, n_boot=n_boot)
    except ValueError as e:
        report["light_minus_dark_gap_ci"] = {"note": str(e)}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    loaders = get_dataloaders(cfg)

    ham = predict(model, loaders["test"], device)
    ham.to_csv(out_dir / "predictions_ham_test.csv", index=False)
    metrics = {"checkpoint": args.checkpoint,
               "ham_test": classification_metrics(ham["label"], ham["pred"], probs_of(ham))}

    if "fitz" in loaders:
        fitz = predict(model, loaders["fitz"], device)
        fitz.to_csv(out_dir / "predictions_fitz.csv", index=False)
        metrics["fitzpatrick17k"] = {
            "overall": classification_metrics(fitz["label"], fitz["pred"], probs_of(fitz)),
            **fairness_report(fitz, args.n_boot),
        }
    else:
        metrics["fitzpatrick17k"] = {"note": "Fitzpatrick17k not found; skin-tone analysis skipped"}

    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"HAM10000 test: bal-acc={metrics['ham_test']['balanced_accuracy']:.3f} "
          f"macro-F1={metrics['ham_test']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
