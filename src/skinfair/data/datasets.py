"""PyTorch datasets for HAM10000 and Fitzpatrick17k.

Owner: Afzal Raza

Every item is a dict:
    image: Tensor[3, H, W], label: int, fitzpatrick: int (-1 if unknown), image_id: str
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from skinfair import CLASSES
from skinfair.data.fitzpatrick_mapping import FITZ_TO_HAM
from skinfair.data.sampling import make_balanced_sampler
from skinfair.data.splits import lesion_level_split
from skinfair.data.transforms import eval_transforms, train_transforms

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    def __init__(self, root: str, df: pd.DataFrame, transform=None):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / "images" / f"{row.image_id}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "label": CLASS_TO_IDX[row.dx],
            "fitzpatrick": -1,  # HAM10000 has no skin-type labels
            "image_id": row.image_id,
        }


class Fitzpatrick17kDataset(Dataset):
    """Evaluation-only: the subset of Fitzpatrick17k whose `label` maps onto a
    HAM10000 class (see fitzpatrick_mapping.FITZ_TO_HAM), with its Fitzpatrick
    skin type (1-6, or -1 if unknown/not recorded).
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        df = pd.read_csv(self.root / "fitzpatrick17k.csv")
        df["dx"] = df["label"].str.lower().str.strip().map(FITZ_TO_HAM)
        self.df = df.dropna(subset=["dx"]).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / "images" / f"{row.md5hash}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        scale = row.fitzpatrick_scale
        return {
            "image": img,
            "label": CLASS_TO_IDX[row.dx],
            "fitzpatrick": int(scale) if pd.notna(scale) and scale >= 1 else -1,
            "image_id": row.md5hash,
        }


def get_dataloaders(cfg):
    """Build train/val/test loaders over HAM10000, plus an eval-only loader
    over the HAM-mappable subset of Fitzpatrick17k (None if fitz_root isn't
    configured or doesn't exist yet).
    """
    d = cfg["data"]
    ham_root = Path(d["ham_root"])
    meta = pd.read_csv(ham_root / "HAM10000_metadata.csv")
    train_df, val_df, test_df = lesion_level_split(meta, seed=cfg.get("seed", 42))

    train_tf = train_transforms(d["image_size"])
    eval_tf = eval_transforms(d["image_size"])

    train_ds = HAM10000Dataset(ham_root, train_df, transform=train_tf)
    val_ds = HAM10000Dataset(ham_root, val_df, transform=eval_tf)
    test_ds = HAM10000Dataset(ham_root, test_df, transform=eval_tf)

    loader_kwargs = {"batch_size": d["batch_size"], "num_workers": d["num_workers"]}

    if d.get("oversample"):
        train_labels = train_df["dx"].map(CLASS_TO_IDX).to_numpy()
        sampler = make_balanced_sampler(train_labels)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    fitz_loader = None
    fitz_root = d.get("fitz_root")
    if fitz_root and (Path(fitz_root) / "fitzpatrick17k.csv").exists():
        fitz_ds = Fitzpatrick17kDataset(fitz_root, transform=eval_tf)
        if len(fitz_ds) > 0:
            fitz_loader = DataLoader(fitz_ds, shuffle=False, **loader_kwargs)

    return {"train": train_loader, "val": val_loader, "test": test_loader, "fitz": fitz_loader}
