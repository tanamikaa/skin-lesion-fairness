"""Generate explanation maps + quality scores for a checkpoint.

Usage:
  python scripts/explain.py --config configs/resnet50.yaml --checkpoint results/checkpoints/resnet50_best.pt
  python scripts/explain.py --config configs/vit_b16.yaml  --checkpoint results/checkpoints/vit_b16_best.pt

Owners: Anamika (Grad-CAM), Khushi (attention rollout), Karanpal (quality metrics).

Writes to <output_dir>/<run_name>/explain/<split>/:
  maps_<method>.npz        image_id + heat-maps (float16, [N, 224, 224]) for the first --max-images images
  quality_<method>.csv     per-image deletion / insertion AUC (+ fitzpatrick)
  overlay_<method>.png     grid of overlays (also copied to reports/figures/)
Methods: ResNet -> gradcam ; ViT -> attention_rollout (+ gradcam_vit for a like-for-like comparison).
--split test|fitz selects the images (fitz gives per-skin-tone quality).

Arguments:
  --config      YAML config file (required)
  --checkpoint  trained model weights file (required)
  --split       "test" (HAM10000 test) or "fitz" (Fitzpatrick17k); default "test"
  --max-images  stop after about this many images (default 256)
  --steps       number of steps for the deletion / insertion curves (default 20)
  --set         optional KEY=VALUE config overrides
Note: the overlay file name actually used by the code is overlay_<run>_<method>_<split>.png.
"""
import argparse
import logging
from pathlib import Path

import matplotlib

# "Agg" backend draws to files only (no window), so it works on servers without a screen.
# It must be set before pyplot is imported, hence the "noqa: E402" on the imports below.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from skinfair.data.datasets import get_dataloaders  # noqa: E402
from skinfair.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from skinfair.explain.attention import attention_rollout, gradcam_vit  # noqa: E402
from skinfair.explain.gradcam import gradcam  # noqa: E402
from skinfair.explain.quality import deletion_auc, insertion_auc  # noqa: E402
from skinfair.models import build_model, load_checkpoint  # noqa: E402
from skinfair.utils import apply_overrides  # noqa: E402


def denorm(x):
    """Undo ImageNet normalisation so an image tensor can be shown.

    Input: x - one image tensor of shape [3, H, W], normalised with ImageNet mean/std.
    Returns: a numpy array [H, W, 3] with values in 0..1, ready for plt.imshow.
    """
    # Reshape mean/std to [3, 1, 1] so they broadcast over each colour channel.
    m = torch.tensor(IMAGENET_MEAN)[:, None, None]
    s = torch.tensor(IMAGENET_STD)[:, None, None]
    # x * std + mean reverses normalisation; clamp to 0..1; permute moves channels last.
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def save_overlays(images, maps, path, n=8, title=""):
    """Save a 2-row figure: original images on top, images with heat-map overlay below.

    Inputs: images - normalised image tensors [N, 3, H, W]; maps - heat-maps [N, H, W] in 0..1;
            path - output PNG path; n - max number of columns; title - figure title.
    Returns nothing (writes the PNG file).
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    # With n == 1, subplots returns a 1-D array; this forces a 2 x n grid either way.
    axes = np.atleast_2d(axes).reshape(2, n)
    for i in range(n):
        img = denorm(images[i])
        axes[0, i].imshow(img)
        axes[1, i].imshow(img)
        # Draw the heat-map semi-transparent (alpha) on top of the image, colours fixed to 0..1.
        axes[1, i].imshow(maps[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, i].axis("off")  # hide axis ticks and frame
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)  # free memory; figures stay open otherwise


def main():
    """Compute explanation maps and their quality scores for one checkpoint.

    Reads the command-line arguments (see module docstring), loads the model, runs each
    explanation method on batches of the chosen split, scores each map with deletion /
    insertion AUC, saves maps (.npz), scores (.csv) and overlay figures (.png), and prints the
    mean scores per method. Returns nothing.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["test", "fitz"])
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    # Config device, else GPU if available, else CPU.
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint, not ImageNet
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    # Run name from the checkpoint, or the file name without "_best".
    run = ckpt.get("run_name") or Path(args.checkpoint).stem.removesuffix("_best")
    model.eval()  # inference mode (no dropout, fixed batch-norm)

    # Choose explanation methods by model type: ViT gets two methods, ResNet gets Grad-CAM.
    is_vit = cfg["model"]["name"].startswith("vit")
    methods = {"attention_rollout": attention_rollout, "gradcam_vit": gradcam_vit} if is_vit else {"gradcam": gradcam}

    loader = get_dataloaders(cfg)[args.split]  # DataLoader for "test" or "fitz"
    out_dir = Path(cfg["output_dir"]) / run / "explain" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each method, collect heat-map batches and per-image score tables.
    store = {m: {"maps": [], "rows": []} for m in methods}
    first_images, seen = None, 0  # first batch is kept for the overlay figure
    for batch in loader:
        # Stop once enough images are processed (the last batch may go a bit over the limit).
        if seen >= args.max_images:
            break
        x = batch["image"].to(device)
        with torch.no_grad():
            # Predicted class per image; maps explain this predicted class.
            tgt = model(x).argmax(1)
        if first_images is None:
            first_images = x.cpu()
        for name, fn in methods.items():
            # attention_rollout is class-agnostic, so it gets no targets; the others do.
            maps = fn(model, x) if name == "attention_rollout" else fn(model, x, targets=tgt)
            # Deletion AUC: remove most important pixels first; a fast drop is good (lower better).
            d = deletion_auc(model, x, maps, steps=args.steps, targets=tgt)
            # Insertion AUC: add most important pixels first; a fast rise is good (higher better).
            i = insertion_auc(model, x, maps, steps=args.steps, targets=tgt)
            # float16 halves the file size of the saved maps.
            store[name]["maps"].append(maps.numpy().astype(np.float16))
            # One row per image in this batch.
            store[name]["rows"].append(pd.DataFrame({
                "image_id": list(batch["image_id"]), "fitzpatrick": batch["fitzpatrick"].numpy(),
                "label": batch["label"].numpy(), "pred": tgt.cpu().numpy(),
                "deletion_auc": d, "insertion_auc": i,
            }))
        seen += len(x)

    # Save results for each method.
    for name, s in store.items():
        maps = np.concatenate(s["maps"])  # join all batches into one [N, H, W] array
        q = pd.concat(s["rows"], ignore_index=True)
        np.savez_compressed(out_dir / f"maps_{name}.npz", image_id=q["image_id"].to_numpy(), maps=maps)
        q.to_csv(out_dir / f"quality_{name}.csv", index=False)
        # Save the same overlay figure in the run folder and in reports/figures.
        # Only the first batch is used, and save_overlays shows at most 8 images.
        for dest in (out_dir, Path("reports/figures")):
            save_overlays(first_images, maps[: len(first_images)].astype(np.float32),
                          dest / f"overlay_{run}_{name}_{args.split}.png",
                          title=f"{cfg['model']['name']} - {name}")
        print(f"{name}: deletion AUC {q['deletion_auc'].mean():.3f} (lower better), "
              f"insertion AUC {q['insertion_auc'].mean():.3f} (higher better), n={len(q)}")


if __name__ == "__main__":
    main()
