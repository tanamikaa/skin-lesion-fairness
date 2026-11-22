# ---------------------------------------------------------------------------
# explain.py: makes explanation heat-maps and quality scores for a checkpoint.
#
# Usage:
#   python scripts/explain.py --config configs/resnet50.yaml --checkpoint results/checkpoints/resnet50_best.pt
#   python scripts/explain.py --config configs/vit_b16.yaml  --checkpoint results/checkpoints/vit_b16_best.pt
#
# What it does: for a trained model it shows WHERE in the image the model looked.
# Methods: ResNet -> gradcam; ViT -> attention_rollout (+ gradcam_vit for a
# like-for-like comparison). Then it scores each map with deletion / insertion AUC.
# It writes to <output_dir>/<run_name>/explain/<split>/:
#   maps_<method>.npz       image_id + heat-maps (float16, [N, 224, 224]) for the first --max-images images
#   quality_<method>.csv    per-image deletion / insertion AUC (+ fitzpatrick)
#   overlay_<run>_<method>_<split>.png   grid of overlays (also copied to reports/figures/)
#
# Where it fits in the pipeline (data -> model -> train -> evaluate -> explain):
# this is the "explain" step, run after training.
# ---------------------------------------------------------------------------

# Lets us read options typed on the command line.
import argparse
# Lets us print progress messages.
import logging
# Path makes file and folder paths easy to handle.
from pathlib import Path

# matplotlib draws the pictures.
import matplotlib

# "Agg" draws to files only (no window), so it works without a screen.
# It must be set before pyplot is imported, so the imports below carry "noqa: E402".
matplotlib.use("Agg")
# pyplot is the plotting interface.
import matplotlib.pyplot as plt  # noqa: E402
# numpy does array maths.
import numpy as np  # noqa: E402
# pandas builds the score tables.
import pandas as pd  # noqa: E402
# PyTorch runs the model.
import torch  # noqa: E402
# Reads YAML config files.
import yaml  # noqa: E402

# Builds the data loaders for the datasets.
from skinfair.data.datasets import get_dataloaders  # noqa: E402
# The mean and std used to normalise images (needed to undo it).
from skinfair.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
# Explanation methods for the ViT model.
from skinfair.explain.attention import attention_rollout, gradcam_vit  # noqa: E402
# Grad-CAM explanation method for the ResNet model.
from skinfair.explain.gradcam import gradcam  # noqa: E402
# Scores that tell how good an explanation map is.
from skinfair.explain.quality import deletion_auc, insertion_auc  # noqa: E402
# Helpers to build the model and load saved weights.
from skinfair.models import build_model, load_checkpoint  # noqa: E402
# Helper that changes config values using KEY=VALUE text.
from skinfair.utils import apply_overrides  # noqa: E402


# denorm: turns a normalised image back into a normal picture.
# Input: x, a [3, H, W] tensor that was normalised with the ImageNet mean and std.
# Returns: a [H, W, 3] numpy array with values in 0..1, ready for imshow.
# Why it exists: the model sees normalised numbers, which look wrong if plotted directly.
def denorm(x):
    """
    This function turns a normalised image back into a normal picture.
    Input: x, an image tensor [3, H, W] that was normalised with ImageNet mean and std.
    It reshapes the mean and std so they apply to each colour channel.
    It multiplies by the std, adds the mean, and keeps values between 0 and 1.
    It moves the channels last and converts to a numpy array.
    It returns an array [H, W, 3] ready for imshow.
    It is needed so the plotted images look natural to a human.
    """
    # Shape the mean as [3, 1, 1] so it applies to each colour channel.
    m = torch.tensor(IMAGENET_MEAN)[:, None, None]
    # Do the same reshape for the std.
    s = torch.tensor(IMAGENET_STD)[:, None, None]
    # Reverse the normalising, keep values in 0..1, and move channels last for imshow.
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# save_overlays: saves a picture grid of images and their heat-map overlays.
# Inputs:
#   images: batch of normalised image tensors.
#   maps: heat-maps (one 2-D array per image, values 0..1).
#   path: Path of the PNG file to write.
#   n: maximum number of images to show (default 8).
#   title: title text for the whole figure.
# Returns: nothing; it writes a PNG with 2 rows (original on top, overlay below).
# Why it exists: a picture lets a human see what the model focused on.
def save_overlays(images, maps, path, n=8, title=""):
    """
    This function saves a picture grid of images and their heatmaps.
    Inputs: images, maps (one heatmap each), path to save, n images to show, and a title.
    It uses at most n images and makes a grid with 2 rows.
    The top row shows the original image; the bottom row shows the heatmap over it.
    It turns off the axes, adds the title, and creates the folder if needed.
    It saves the figure to path and closes it; nothing is returned.
    It is needed to see what parts of the image the model looked at.
    """
    # Show at most n images, or fewer if we have fewer.
    n = min(n, len(images))
    # Make a grid with 2 rows and n columns.
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    # If n is 1, axes is 1-D; this makes it a 2 x n grid either way.
    axes = np.atleast_2d(axes).reshape(2, n)
    # Go through each column.
    for i in range(n):
        # Get a viewable version of this image.
        img = denorm(images[i])
        # Top row: the plain image.
        axes[0, i].imshow(img)
        # Bottom row: the plain image first.
        axes[1, i].imshow(img)
        # Draw the heat-map see-through on top, with colours fixed to 0..1.
        axes[1, i].imshow(maps[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        # Do this for both rows.
        for r in (0, 1):
            # Hide the axis ticks and frame.
            axes[r, i].axis("off")
    # Put the title on top.
    fig.suptitle(title)
    # Fix the spacing so nothing overlaps.
    fig.tight_layout()
    # Make sure the output folder exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the figure as a PNG.
    fig.savefig(path, dpi=120)
    # Close the figure to free memory.
    plt.close(fig)


# main: runs the whole explanation job for one checkpoint.
# Inputs: none directly; the options are read from the command line (see the top of the file).
# Returns: nothing. It saves the maps, score CSVs and overlay pictures, and prints mean scores.
# Why it exists: it is the command-line entry point of the explain step.
def main():
    """
    This function is the entry point of the explanation script.
    Input: command-line options such as --config, --checkpoint, --split and --max-images.
    It builds the model, loads the weights, and picks the explain methods for ResNet or ViT.
    For each batch it makes heatmaps and scores them with deletion and insertion AUC.
    It saves the maps (npz), the quality scores (csv) and overlay pictures.
    It returns nothing; it prints the average scores for each method.
    It is needed to check if the model looks at the lesion, for every skin tone.
    """
    # Create the object that reads command-line options.
    p = argparse.ArgumentParser()
    # YAML file with all settings (must be given).
    p.add_argument("--config", required=True)
    # Saved model weights (must be given).
    p.add_argument("--checkpoint", required=True)
    # Which images to explain: HAM10000 test or Fitzpatrick17k.
    p.add_argument("--split", default="test", choices=["test", "fitz"])
    # Stop after about this many images.
    p.add_argument("--max-images", type=int, default=256)
    # Number of steps in the deletion and insertion curves.
    p.add_argument("--steps", type=int, default=20)
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
    # Do not download pretrained weights; they come from the checkpoint.
    cfg["model"]["pretrained"] = False
    # Build the model and move it to the device.
    model = build_model(cfg).to(device)
    # Load the saved weights into the model.
    ckpt = load_checkpoint(model, args.checkpoint, device)
    # Run name from the checkpoint, or else from the file name without "_best".
    run = ckpt.get("run_name") or Path(args.checkpoint).stem.removesuffix("_best")
    # Switch to evaluation mode (no dropout, fixed batch-norm).
    model.eval()

    # Check if this is a ViT model (its name starts with "vit").
    is_vit = cfg["model"]["name"].startswith("vit")
    # ViT gets two explanation methods; ResNet gets Grad-CAM.
    methods = {"attention_rollout": attention_rollout, "gradcam_vit": gradcam_vit} if is_vit else {"gradcam": gradcam}

    # Data loader for the chosen split ("test" or "fitz").
    loader = get_dataloaders(cfg)[args.split]
    # Folder where all outputs are saved.
    out_dir = Path(cfg["output_dir"]) / run / "explain" / args.split
    # Create the folder if it does not exist yet.
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each method, make lists to hold the heat-maps and the score tables.
    store = {m: {"maps": [], "rows": []} for m in methods}
    # first_images keeps the first batch for the overlay picture; seen counts images done so far.
    first_images, seen = None, 0
    # Go through the data batch by batch.
    for batch in loader:
        # Stop when we have enough images (the last batch may go a little over).
        if seen >= args.max_images:
            # Leave the loop.
            break
        # Move the images to the device.
        x = batch["image"].to(device)
        # No gradients are needed just to get predictions.
        with torch.no_grad():
            # The predicted class for each image; the maps explain this class.
            tgt = model(x).argmax(1)
        # On the first batch only.
        if first_images is None:
            # Save a CPU copy for the overlay picture.
            first_images = x.cpu()
        # Run each explanation method.
        for name, fn in methods.items():
            # attention_rollout does not use a target class; the other methods do.
            maps = fn(model, x) if name == "attention_rollout" else fn(model, x, targets=tgt)
            # Deletion AUC: hide the most important pixels first; lower is better.
            d = deletion_auc(model, x, maps, steps=args.steps, targets=tgt)
            # Insertion AUC: reveal the most important pixels first; higher is better.
            i = insertion_auc(model, x, maps, steps=args.steps, targets=tgt)
            # Store the maps as float16 to halve the file size.
            store[name]["maps"].append(maps.numpy().astype(np.float16))
            # Store one table row per image in this batch
            # (image id, skin type, true label, predicted class, and the two AUC scores).
            store[name]["rows"].append(pd.DataFrame({
                "image_id": list(batch["image_id"]), "fitzpatrick": batch["fitzpatrick"].numpy(),
                "label": batch["label"].numpy(), "pred": tgt.cpu().numpy(),
                "deletion_auc": d, "insertion_auc": i,
            }))
        # Count the images we just did.
        seen += len(x)

    # Save the results for each method.
    for name, s in store.items():
        # Join all batches into one [N, H, W] array.
        maps = np.concatenate(s["maps"])
        # Join all the score tables into one.
        q = pd.concat(s["rows"], ignore_index=True)
        # Save the image ids and maps in one compressed file.
        np.savez_compressed(out_dir / f"maps_{name}.npz", image_id=q["image_id"].to_numpy(), maps=maps)
        # Save the per-image scores as a CSV.
        q.to_csv(out_dir / f"quality_{name}.csv", index=False)
        # Save the same overlay picture in the run folder and in reports/figures.
        # Only the first batch is used, and at most 8 images are shown.
        for dest in (out_dir, Path("reports/figures")):
            # Draw and save the overlay grid for this folder.
            save_overlays(first_images, maps[: len(first_images)].astype(np.float32),
                          dest / f"overlay_{run}_{name}_{args.split}.png",
                          title=f"{cfg['model']['name']} - {name}")
        # Print the average scores for this method.
        print(f"{name}: deletion AUC {q['deletion_auc'].mean():.3f} (lower better), "
              f"insertion AUC {q['insertion_auc'].mean():.3f} (higher better), n={len(q)}")


# Run main() only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Start the explanation run.
    main()
