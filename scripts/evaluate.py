# ---------------------------------------------------------------------------
# evaluate.py: evaluates a trained model.
#
# Usage:
#   python scripts/evaluate.py --config ... --checkpoint ...
#
# What it does: after training, it asks the saved model for predictions on two datasets
# (the HAM10000 test split and Fitzpatrick17k) and turns them into numbers: overall
# scores on HAM10000, and fairness scores on Fitzpatrick17k (light vs dark skin).
# It writes to <output_dir>/<run_name>/:
#   predictions_ham_test.csv, predictions_fitz.csv, metrics.json
#   (overall scores, per skin type scores, gaps and confidence intervals).
#
# Fitzpatrick17k is a different domain (clinical photos, fewer classes), so read it as
# a skin-tone robustness probe, not as test-set performance.
#
# Where it fits in the pipeline (data -> model -> train -> evaluate -> explain):
# this is the "evaluate" step. It reads the checkpoint saved by train, and its CSV
# files are then used by compare_models.py and the explain step.
#
# Output file notes:
#   - predictions CSV columns: image_id, label, pred, prob_0..prob_6, fitzpatrick
#     (fitzpatrick = -1 means the skin type is unknown).
#   - metrics.json uses null where a number could not be computed (NaN is not valid JSON).
# ---------------------------------------------------------------------------

# Lets us read options typed on the command line.
import argparse
# Used to write the metrics as JSON.
import json
# Lets us print progress messages.
import logging
# Used to check for NaN (not a number) values.
import math
# Path makes file and folder paths easy to handle.
from pathlib import Path

# PyTorch: used here to pick the device (GPU or CPU).
import torch
# Reads YAML config files.
import yaml

# Builds the data loaders for the datasets.
from skinfair.data.datasets import get_dataloaders
# Fairness helpers that measure differences between skin types.
from skinfair.evaluation.fairness import (
    bootstrap_gap_ci,
    bootstrap_group_ci,
    fairness_gap,
    light_vs_dark,
    per_class_recall_light_vs_dark,
    per_group_metrics,
)
# Computes overall scores like balanced accuracy and F1.
from skinfair.evaluation.metrics import classification_metrics
# predict runs the model; probs_of pulls out the probability columns.
from skinfair.evaluation.predict import predict, probs_of
# Helpers to build the model and load saved weights.
from skinfair.models import build_model, load_checkpoint
# Helper that changes config values using KEY=VALUE text.
from skinfair.utils import apply_overrides


# _clean: makes any object safe to save as JSON.
# Input: o, any nested mix of dicts, lists, numpy numbers and floats.
# Returns: the same structure using only types that json.dumps accepts
#   (NaN becomes None, numpy numbers become plain Python numbers, dict keys become strings).
# Why it exists: json.dumps fails on numpy numbers and writes NaN, which many JSON readers reject.
def _clean(o):
    """
    This function makes a value safe to save as JSON.
    Input: o, any value such as a dict, list, number or numpy item.
    For a dict it cleans every value and turns the keys into strings.
    For a list or tuple it cleans each item and returns a plain list.
    Numpy numbers are turned into normal Python numbers using item().
    A float NaN becomes None; the cleaned value is returned.
    It is needed because JSON cannot store numpy types or NaN.
    """
    # If it is a dict, clean what is inside it.
    if isinstance(o, dict):
        # Clean every value; JSON keys must be strings.
        return {str(k): _clean(v) for k, v in o.items()}
    # If it is a list or tuple, clean each item.
    if isinstance(o, (list, tuple)):
        # Return a plain list of cleaned items.
        return [_clean(v) for v in o]
    # numpy or torch numbers have an .item() method.
    if hasattr(o, "item"):
        # Turn it into a normal Python number.
        o = o.item()
    # JSON has no NaN, so check for it.
    if isinstance(o, float) and math.isnan(o):
        # Use None, which becomes null in the file.
        return None
    # Everything else is already fine.
    return o


# fairness_report: builds all the fairness numbers for one prediction table.
# Inputs:
#   df: predictions table, one row per image (columns label, pred, fitzpatrick; -1 = unknown skin type).
#   n_boot: number of bootstrap redraws used for the confidence intervals.
# Returns: a dict with the keys n_with_fst, per_fst (score + CI per skin type),
#   gap_max_min (best minus worst skin type and their ratio), light_vs_dark (scores of
#   the two merged groups), per_class_recall_light_vs_dark (which classes are missed
#   more on dark skin) and light_minus_dark_gap_ci (the gap with its 95% CI).
#   Parts that cannot be computed hold {"note": <error message>}.
# Why it exists: this one dict is what the report's fairness section is built from.
def fairness_report(df, n_boot):
    """
    This function measures how fair the model is across skin types.
    Inputs: df, a table of predictions with a fitzpatrick column; n_boot, the resample count.
    It keeps rows with a known skin type and stops with a note if none exist.
    Then it computes per-type accuracy with confidence intervals and the max-min gap.
    It also compares light and dark skin, per-class recall, and the gap interval.
    It returns one dictionary with all these results.
    It is needed to show if the model works equally well for all skin tones.
    """
    # Keep only rows whose skin type is known (1 to 6).
    known = df[df["fitzpatrick"] >= 1]
    # If no row has a known skin type, we cannot measure anything.
    if known.empty:
        # Return a note that explains why.
        return {"note": "no rows with a known Fitzpatrick type"}
    # Balanced accuracy for each skin type (unknown rows are ignored inside).
    per_fst = per_group_metrics(df)
    # Start the report dict.
    # "n_with_fst": how many rows had a known skin type.
    # "per_fst": per skin type scores with confidence intervals, as a list of row dicts.
    # "gap_max_min": gap between the best and the worst skin type.
    # "light_vs_dark": scores for light (I-III) vs dark (IV-VI) skin.
    report = {
        "n_with_fst": int(len(known)),
        "per_fst": bootstrap_group_ci(df, n_boot=n_boot).to_dict("records"),
        "gap_max_min": fairness_gap(per_fst),
        "light_vs_dark": light_vs_dark(df).to_dict("records"),
    }
    # These two parts can fail (e.g. no light or no dark rows gives a ValueError), so catch
    # the error and store a note instead of crashing the whole evaluation.
    try:
        # Recall for each disease class, light vs dark.
        report["per_class_recall_light_vs_dark"] = per_class_recall_light_vs_dark(df).to_dict("records")
    # If that failed, handle the error here.
    except ValueError as e:
        # Keep the reason in the report instead.
        report["per_class_recall_light_vs_dark"] = {"note": str(e)}
    # Second part that can fail.
    try:
        # Light-minus-dark gap with a confidence interval.
        report["light_minus_dark_gap_ci"] = bootstrap_gap_ci(df, n_boot=n_boot)
    # If that failed, handle the error here.
    except ValueError as e:
        # Keep the reason in the report instead.
        report["light_minus_dark_gap_ci"] = {"note": str(e)}
    # Give back the finished report.
    return report


# main: runs the whole evaluation of one checkpoint.
# Inputs: none directly; the options are read from the command line (see the top of the file).
# Returns: nothing. It saves the prediction CSVs and metrics.json, and prints HAM10000 scores.
# Steps: read options and config -> build model and load weights -> predict on the
#   HAM10000 test set -> predict on Fitzpatrick17k (if available) -> compute metrics and
#   fairness numbers -> save everything under <output_dir>/<run_name>/.
def main():
    """
    This function is the entry point of the evaluation script.
    Input: command-line options such as --config, --checkpoint and --n-boot.
    It reads the config, builds the model and loads the saved weights.
    It predicts on the HAM10000 test set and on Fitzpatrick17k if it exists.
    It computes metrics and the fairness report, and saves CSV and JSON files.
    It returns nothing; it prints the output path and the main scores.
    It is needed to test a trained model and record how well and how fairly it works.
    """
    # Create the object that reads command-line options.
    p = argparse.ArgumentParser()
    # YAML file with all settings (must be given).
    p.add_argument("--config", required=True)
    # Saved model weights (must be given).
    p.add_argument("--checkpoint", required=True)
    # Number of bootstrap resamples for the confidence intervals.
    p.add_argument("--n-boot", type=int, default=1000)
    # Any number of KEY=VALUE config changes.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    # Read the options the user typed and store them in args.
    args = p.parse_args()
    # Show INFO messages while running.
    logging.basicConfig(level=logging.INFO)

    # Open the config file for reading.
    with open(args.config) as f:
        # Load the YAML and apply the --set changes.
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    # Use the config's device if set, else the GPU if there is one, else the CPU.
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Do not download pretrained weights; the checkpoint already holds the trained
    # weights, so a download would be wasted.
    cfg["model"]["pretrained"] = False
    # Build the model and move it to the device.
    model = build_model(cfg).to(device)
    # Load the saved weights into the model.
    ckpt = load_checkpoint(model, args.checkpoint, device)
    # Run name from the checkpoint, or else from the file name without "_best".
    # e.g. "resnet50_focal_s42_best.pt" -> "resnet50_focal_s42".
    name = ckpt.get("run_name") or Path(args.checkpoint).stem.removesuffix("_best")
    # Output folder, the same one training used.
    out_dir = Path(cfg["output_dir"]) / name
    # Create the folder if it does not exist yet.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dict of data loaders, e.g. "test" and maybe "fitz".
    loaders = get_dataloaders(cfg)

    # Predict on the HAM10000 test split (one row per image).
    ham = predict(model, loaders["test"], device)
    # Save those predictions to a CSV.
    ham.to_csv(out_dir / "predictions_ham_test.csv", index=False)
    # Start the metrics dict with the overall HAM10000 scores
    # (probs_of gets the probability array, which is needed for AUROC).
    metrics = {"checkpoint": args.checkpoint,
               "ham_test": classification_metrics(ham["label"], ham["pred"], probs_of(ham))}

    # The "fitz" loader exists only if the Fitzpatrick17k data was found.
    if "fitz" in loaders:
        # Predict on the Fitzpatrick17k images.
        fitz = predict(model, loaders["fitz"], device)
        # Save those predictions to a CSV.
        fitz.to_csv(out_dir / "predictions_fitz.csv", index=False)
        # Store the overall scores plus the fairness numbers.
        # The ** copies the fairness report's keys into this dict.
        metrics["fitzpatrick17k"] = {
            "overall": classification_metrics(fitz["label"], fitz["pred"], probs_of(fitz)),
            **fairness_report(fitz, args.n_boot),
        }
    # Otherwise there is no Fitzpatrick data.
    else:
        # Leave a note saying the skin-tone part was skipped.
        metrics["fitzpatrick17k"] = {"note": "Fitzpatrick17k not found; skin-tone analysis skipped"}

    # Make everything JSON-safe and save it as readable JSON.
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    # Tell the user where the file went.
    print(f"Wrote {out_dir / 'metrics.json'}")
    # Print the two main HAM10000 scores.
    print(f"HAM10000 test: bal-acc={metrics['ham_test']['balanced_accuracy']:.3f} "
          f"macro-F1={metrics['ham_test']['macro_f1']:.3f}")


# Run main() only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Start the evaluation.
    main()
