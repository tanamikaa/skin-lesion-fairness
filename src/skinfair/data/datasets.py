"""PyTorch datasets for HAM10000 and Fitzpatrick17k.

Owner: Afzal Raza

Every item is a dict:
    image: Tensor[3, H, W], label: int, fitzpatrick: int (-1 if unknown), image_id: str
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from skinfair import CLASSES

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
    """TODO(Afzal): load CSV, keep only rows mappable to HAM classes, return FST label."""

    def __init__(self, root: str, transform=None):
        raise NotImplementedError


def get_dataloaders(cfg):
    """TODO(Afzal): return dict(train=..., val=..., test=..., fitz=...) DataLoaders."""
    raise NotImplementedError
