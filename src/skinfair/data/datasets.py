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

# class name -> integer label, e.g. {"akiec": 0, "bcc": 1, ..., "vasc": 6}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_ham_metadata(root: str) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and validate the diagnosis column.

    Input: dataset root folder. Returns the metadata DataFrame (one row per image, with
    columns such as image_id, lesion_id, dx). Raises ValueError on an unknown diagnosis.
    """
    meta = pd.read_csv(Path(root) / "HAM10000_metadata.csv")
    unknown = set(meta["dx"]) - set(CLASSES)  # diagnoses in the CSV that are not in CLASSES
    if unknown:
        raise ValueError(f"Unknown HAM10000 diagnoses: {sorted(unknown)}")
    return meta


class HAM10000Dataset(Dataset):
    """HAM10000 images for one split (train, val or test).

    Each item is a dict: image (transformed tensor), label (int), fitzpatrick (-1),
    image_id (str). Images are read from <root>/images/<image_id>.jpg.
    """

    def __init__(self, root: str, df: pd.DataFrame, transform=None):
        """Inputs: root folder, metadata rows of this split, optional image transform."""
        self.root = Path(root)
        self.df = df.reset_index(drop=True)  # index 0..n-1 so iloc[idx] matches idx
        self.transform = transform

    @property
    def labels(self):
        """NumPy array of integer labels, one per image (used for class counts / sampler)."""
        return self.df["dx"].map(CLASS_TO_IDX).to_numpy()

    def __len__(self):
        """Number of images in this split."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load image number `idx` and return it as a dict (see the class docstring)."""
        row = self.df.iloc[idx]
        # convert("RGB") guarantees 3 channels (e.g. if a file were grayscale or RGBA)
        img = Image.open(self.root / "images" / f"{row.image_id}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)  # PIL image -> normalised tensor [3, H, W]
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
        """Read the CSV, keep only mappable rows with a downloaded image, clean `fst`.

        Inputs: root folder, optional image transform, CSV file name.
        """
        self.root = Path(root)
        self.transform = transform
        df = pd.read_csv(self.root / csv_name)
        df["dx"] = df["label"].map(map_label)  # HAM class name, or None if not mappable
        n_all = len(df)
        df = df[df["dx"].notna()]  # drop conditions with no HAM equivalent
        n_mapped = len(df)
        # True/False per row: does <root>/images/<md5hash>.jpg exist on disk?
        has_img = df["md5hash"].map(lambda h: (self.root / "images" / f"{h}.jpg").exists())
        df = df[has_img]
        if len(df) < n_mapped:
            log.warning("Fitzpatrick17k: %d/%d mapped images missing on disk", n_mapped - len(df), n_mapped)
        log.info("Fitzpatrick17k: kept %d of %d rows (mappable + downloaded)", len(df), n_all)
        df = df.rename(columns={"md5hash": "image_id", "fitzpatrick_scale": "fst"})
        # keep skin types 1-6; any other value (unknown code or missing) becomes -1
        df["fst"] = df["fst"].where(df["fst"].between(1, 6), -1).astype(int)
        self.df = df.reset_index(drop=True)

    @property
    def labels(self):
        """NumPy array of integer HAM labels, one per kept image."""
        return self.df["dx"].map(CLASS_TO_IDX).to_numpy()

    def __len__(self):
        """Number of kept images."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load image `idx`; returns dict(image, label, fitzpatrick (1-6 or -1), image_id)."""
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
    HAM10000 is split into train/val/test; Fitzpatrick17k is only used as an extra
    evaluation set (no training on it).
    """
    d = cfg["data"]
    seed = cfg.get("seed", 42)
    size = d.get("image_size", 224)  # images are resized/cropped to size x size
    nw = d.get("num_workers", 0)  # 0 = load data in the main process

    meta = load_ham_metadata(d["ham_root"])
    train_df, val_df, test_df = lesion_level_split(meta, seed=seed)
    assert_no_leakage(train_df, val_df, test_df)  # stop if a lesion is in two splits

    # augmentation only for training; val/test get the deterministic eval transform
    train_ds = HAM10000Dataset(d["ham_root"], train_df, train_transforms(size))
    val_ds = HAM10000Dataset(d["ham_root"], val_df, eval_transforms(size))
    test_ds = HAM10000Dataset(d["ham_root"], test_df, eval_transforms(size))

    common = dict(  # DataLoader settings shared by every loader
        batch_size=d.get("batch_size", 32),
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),  # page-locked memory speeds up CPU->GPU copies
        persistent_workers=nw > 0,  # keep workers alive between epochs (only valid if nw > 0)
    )
    if d.get("oversample", False):
        # sampler draws rare classes more often; DataLoader forbids shuffle with a sampler
        sampler = make_balanced_sampler(train_ds.labels, num_classes=len(CLASSES), seed=seed)
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
    else:
        g = torch.Generator().manual_seed(seed)  # seeded generator -> repeatable shuffle order
        train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)

    loaders = {
        "train": train_loader,
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }

    fitz_root = d.get("fitz_root")  # optional; may be missing from the config
    if fitz_root and (Path(fitz_root) / "fitzpatrick17k.csv").exists():
        fitz_ds = Fitzpatrick17kDataset(fitz_root, eval_transforms(size))
        loaders["fitz"] = DataLoader(fitz_ds, shuffle=False, **common)
    else:
        log.warning("Fitzpatrick17k CSV not found under %s; 'fitz' loader omitted", fitz_root)
    return loaders
