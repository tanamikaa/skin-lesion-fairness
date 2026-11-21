"""Tiny synthetic HAM10000 / Fitzpatrick17k folders for tests and smoke runs (no real data needed)."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from skinfair import CLASSES

# one Fitzpatrick17k label per HAM class (all present in FITZ_TO_HAM)
FITZ_LABELS = {
    "akiec": "actinic keratosis", "bcc": "basal cell carcinoma", "bkl": "seborrheic keratosis",
    "df": "dermatofibroma", "mel": "melanoma", "nv": "nevocytic nevus", "vasc": "pyogenic granuloma",
}


def _img(path, rng, size=32, tint=0):
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    arr[..., 0] = np.clip(arr[..., 0].astype(int) + tint, 0, 255)
    Image.fromarray(arr).save(path)


def make_synthetic_ham(root, per_class=30, imgs_per_lesion=2, seed=0):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for ci, c in enumerate(CLASSES):
        for i in range(per_class):
            lesion = f"L_{c}_{i // imgs_per_lesion}"
            image_id = f"ISIC_{ci}_{i:03d}"
            _img(root / "images" / f"{image_id}.jpg", rng, tint=ci * 20)
            rows.append({"lesion_id": lesion, "image_id": image_id, "dx": c})
    pd.DataFrame(rows).to_csv(root / "HAM10000_metadata.csv", index=False)
    return root


def make_synthetic_fitz(root, per_class=12, seed=1):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for ci, c in enumerate(CLASSES):
        for i in range(per_class):
            h = f"{ci}{i:02d}abc"
            _img(root / "images" / f"{h}.jpg", rng)
            rows.append({"md5hash": h, "fitzpatrick_scale": (i % 6) + 1, "label": FITZ_LABELS[c].upper(),
                         "url": ""})
    rows.append({"md5hash": "dropme", "fitzpatrick_scale": 3, "label": "psoriasis", "url": ""})  # unmapped
    pd.DataFrame(rows).to_csv(root / "fitzpatrick17k.csv", index=False)
    return root


def make_cfg(tmp_path, model="resnet50", size=32, epochs=2):
    ham = make_synthetic_ham(Path(tmp_path) / "ham")
    fitz = make_synthetic_fitz(Path(tmp_path) / "fitz")
    name = "resnet50" if model == "resnet50" else "vit_base_patch16_224"
    return {
        "seed": 0,
        "model": {"name": name, "pretrained": False, "num_classes": 7, "freeze_backbone_epochs": 1,
                  **({"layer_decay": 0.75} if model != "resnet50" else {})},
        "data": {"ham_root": str(ham), "fitz_root": str(fitz), "image_size": size, "batch_size": 16,
                 "num_workers": 0, "oversample": True},
        "loss": {"name": "focal", "gamma": 2.0},
        "train": {"epochs": epochs, "lr": 1e-3, "weight_decay": 1e-4, "early_stopping_patience": 5,
                  "amp": True, "warmup_epochs": 1, "progress": False},
        "output_dir": str(Path(tmp_path) / "out" / model),
        "checkpoint_dir": str(Path(tmp_path) / "ckpt"),
    }
