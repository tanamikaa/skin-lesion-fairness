"""PyTorch datasets for HAM10000 and Fitzpatrick17k.

Owner: Afzal Raza

Every item is a dict:
    image: Tensor[3, H, W], label: int, fitzpatrick: int (-1 if unknown), image_id: str
"""
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from skinfair import CLASSES

from .fitzpatrick_mapping import map_label
from .sampling import make_balanced_sampler
from .splits import assert_no_leakage, lesion_level_split
from .transforms import eval_transforms, train_transforms

log = logging.getLogger(__name__)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_ham_metadata(root: str) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and validate the diagnosis column."""
    meta = pd.read_csv(Path(root) / "HAM10000_metadata.csv")
    unknown = set(meta["dx"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown HAM10000 diagnoses: {sorted(unknown)}")
    return meta


class HAM10000Dataset(Dataset):
    def __init__(self, root: str, df: pd.DataFrame, transform=None):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    @property
    def labels(self):
        return self.df["dx"].map(CLASS_TO_IDX).to_numpy()

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
    """Fitzpatrick17k restricted to conditions mappable onto the 7 HAM classes.

    Expects `<root>/fitzpatrick17k.csv` (columns: md5hash, fitzpatrick_scale, label, url, ...)
    and images at `<root>/images/<md5hash>.jpg`. Rows whose image has not been downloaded
    are skipped (a warning reports how many), so partial downloads still work.
    `fitzpatrick` is 1-6, or -1 when the CSV says unknown.
    """

    def __init__(self, root: str, transform=None, csv_name="fitzpatrick17k.csv"):
        self.root = Path(root)
        self.transform = transform
        df = pd.read_csv(self.root / csv_name)
        df["dx"] = df["label"].map(map_label)
        n_all = len(df)
        df = df[df["dx"].notna()]
        n_mapped = len(df)
        has_img = df["md5hash"].map(lambda h: (self.root / "images" / f"{h}.jpg").exists())
        df = df[has_img]
        if len(df) < n_mapped:
            log.warning("Fitzpatrick17k: %d/%d mapped images missing on disk", n_mapped - len(df), n_mapped)
        log.info("Fitzpatrick17k: kept %d of %d rows (mappable + downloaded)", len(df), n_all)
        df = df.rename(columns={"md5hash": "image_id", "fitzpatrick_scale": "fst"})
        df["fst"] = df["fst"].where(df["fst"].between(1, 6), -1).astype(int)
        self.df = df.reset_index(drop=True)

    @property
    def labels(self):
        return self.df["dx"].map(CLASS_TO_IDX).to_numpy()

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
            "fitzpatrick": int(row.fst),
            "image_id": row.image_id,
        }


def get_dataloaders(cfg):
    """Return dict(train, val, test[, fitz]) DataLoaders.

    cfg keys used: seed, data.{ham_root, fitz_root, image_size, batch_size, num_workers, oversample}.
    `fitz` is only present when the Fitzpatrick17k CSV exists under data.fitz_root.
    Batches are dict(image, label, fitzpatrick, image_id).
    """
    d = cfg["data"]
    seed = cfg.get("seed", 42)
    size = d.get("image_size", 224)
    nw = d.get("num_workers", 0)

    meta = load_ham_metadata(d["ham_root"])
    train_df, val_df, test_df = lesion_level_split(meta, seed=seed)
    assert_no_leakage(train_df, val_df, test_df)

    train_ds = HAM10000Dataset(d["ham_root"], train_df, train_transforms(size))
    val_ds = HAM10000Dataset(d["ham_root"], val_df, eval_transforms(size))
    test_ds = HAM10000Dataset(d["ham_root"], test_df, eval_transforms(size))

    common = dict(
        batch_size=d.get("batch_size", 32),
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    if d.get("oversample", False):
        sampler = make_balanced_sampler(train_ds.labels, num_classes=len(CLASSES), seed=seed)
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
    else:
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)

    loaders = {
        "train": train_loader,
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }

    fitz_root = d.get("fitz_root")
    if fitz_root and (Path(fitz_root) / "fitzpatrick17k.csv").exists():
        fitz_ds = Fitzpatrick17kDataset(fitz_root, eval_transforms(size))
        loaders["fitz"] = DataLoader(fitz_ds, shuffle=False, **common)
    else:
        log.warning("Fitzpatrick17k CSV not found under %s; 'fitz' loader omitted", fitz_root)
    return loaders
