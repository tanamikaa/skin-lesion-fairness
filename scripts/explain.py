"""Generate explanation maps + quality scores for a checkpoint.

Usage:
  python scripts/explain.py --config configs/resnet50.yaml --checkpoint results/checkpoints/resnet50_best.pt
  python scripts/explain.py --config configs/vit_b16.yaml  --checkpoint results/checkpoints/vit_b16_best.pt

Owners: Anamika (Grad-CAM), Khushi (attention rollout), Karanpal (quality metrics).

Writes to <output_dir>/explain/:
  maps_<method>.npz        image_id + heat-maps (float16, [N, 224, 224]) for the first --max-images images
  quality_<method>.csv     per-image deletion / insertion AUC (+ fitzpatrick)
  overlay_<method>.png     grid of overlays (also copied to reports/figures/)
Methods: ResNet -> gradcam ; ViT -> attention_rollout (+ gradcam_vit for a like-for-like comparison).
--split test|fitz selects the images (fitz gives per-skin-tone quality).
"""
import argparse
import logging
from pathlib import Path

import matplotlib

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
    m = torch.tensor(IMAGENET_MEAN)[:, None, None]
    s = torch.tensor(IMAGENET_STD)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def save_overlays(images, maps, path, n=8, title=""):
    n = min(n, len(images))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    axes = np.atleast_2d(axes).reshape(2, n)
    for i in range(n):
        img = denorm(images[i])
        axes[0, i].imshow(img)
        axes[1, i].imshow(img)
        axes[1, i].imshow(maps[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
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
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg["model"]["pretrained"] = False
    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    is_vit = cfg["model"]["name"].startswith("vit")
    methods = {"attention_rollout": attention_rollout, "gradcam_vit": gradcam_vit} if is_vit else {"gradcam": gradcam}

    loader = get_dataloaders(cfg)[args.split]
    out_dir = Path(cfg["output_dir"]) / "explain" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    store = {m: {"maps": [], "rows": []} for m in methods}
    first_images, seen = None, 0
    for batch in loader:
        if seen >= args.max_images:
            break
        x = batch["image"].to(device)
        with torch.no_grad():
            tgt = model(x).argmax(1)
        if first_images is None:
            first_images = x.cpu()
        for name, fn in methods.items():
            maps = fn(model, x) if name == "attention_rollout" else fn(model, x, targets=tgt)
            d = deletion_auc(model, x, maps, steps=args.steps, targets=tgt)
            i = insertion_auc(model, x, maps, steps=args.steps, targets=tgt)
            store[name]["maps"].append(maps.numpy().astype(np.float16))
            store[name]["rows"].append(pd.DataFrame({
                "image_id": list(batch["image_id"]), "fitzpatrick": batch["fitzpatrick"].numpy(),
                "label": batch["label"].numpy(), "pred": tgt.cpu().numpy(),
                "deletion_auc": d, "insertion_auc": i,
            }))
        seen += len(x)

    for name, s in store.items():
        maps = np.concatenate(s["maps"])
        q = pd.concat(s["rows"], ignore_index=True)
        np.savez_compressed(out_dir / f"maps_{name}.npz", image_id=q["image_id"].to_numpy(), maps=maps)
        q.to_csv(out_dir / f"quality_{name}.csv", index=False)
        for dest in (out_dir, Path("reports/figures")):
            save_overlays(first_images, maps[: len(first_images)].astype(np.float32),
                          dest / f"overlay_{Path(cfg['output_dir']).name}_{name}_{args.split}.png",
                          title=f"{cfg['model']['name']} - {name}")
        print(f"{name}: deletion AUC {q['deletion_auc'].mean():.3f} (lower better), "
              f"insertion AUC {q['insertion_auc'].mean():.3f} (higher better), n={len(q)}")


if __name__ == "__main__":
    main()
