# This file defines the datasets and DataLoaders for HAM10000 and Fitzpatrick17k.
# A Dataset tells PyTorch how to get ONE sample (image + label) by its number.
# A DataLoader groups samples into batches and can load them with several workers.
# HAM10000 is split into train / val / test. Fitzpatrick17k is an extra test set with
# known skin types (1-6), used only to check fairness across skin tones, never for training.
# Each sample is a dict: image (tensor [3, H, W]), label (0-6), fitzpatrick (1-6 or -1),
# and image_id (string).

# logging prints messages and warnings
import logging
# Path makes file paths easy to build
from pathlib import Path

# pandas reads the CSV tables
import pandas as pd
# torch is the main deep learning library
import torch
# PIL opens image files
from PIL import Image
# base classes for datasets and the loader
from torch.utils.data import DataLoader, Dataset

# CLASSES is the list of the 7 class names
from skinfair import CLASSES

# turns a Fitzpatrick17k label into a HAM class
from .fitzpatrick_mapping import map_label
# builds the balanced sampler for oversampling
from .sampling import make_balanced_sampler
# split by lesion, and the leakage check
from .splits import assert_no_leakage, lesion_level_split
# the image pipelines for training and evaluation
from .transforms import eval_transforms, train_transforms

# logger for messages and warnings from this file
log = logging.getLogger(__name__)

# class name -> number, like {"akiec": 0, "bcc": 1, ..., "vasc": 6} (models use numbers)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# Reads the HAM10000 metadata CSV and checks the diagnosis column.
# Args: root: the HAM10000 folder (must contain HAM10000_metadata.csv).
# Returns: a table with one row per image (columns include image_id, lesion_id, dx).
# Why: an unknown diagnosis would break the label lookup later.
def load_ham_metadata(root: str) -> pd.DataFrame:
    """Reads the HAM10000 metadata CSV file and checks it.
    Step 1: open HAM10000_metadata.csv inside the given folder.
    Step 2: find any diagnosis (dx) that is not one of our 7 known classes.
    Step 3: raise an error if such an unknown diagnosis exists.
    Input: root, the path of the HAM10000 folder.
    Output: a table with one row per image. It is needed so label errors are caught early.
    """
    # read the CSV: one row per image
    meta = pd.read_csv(Path(root) / "HAM10000_metadata.csv")
    # diagnoses in the file that we do not know
    unknown = set(meta["dx"]) - set(CLASSES)
    # stop early if there is a surprise class
    if unknown:
        raise ValueError(f"Unknown HAM10000 diagnoses: {sorted(unknown)}")
    # give back the table
    return meta


# Dataset of HAM10000 images for one split (train, val or test).
# A Dataset must offer __len__ (how many samples) and __getitem__ (get sample number i).
class HAM10000Dataset(Dataset):
    # Saves the folder, the rows of this split and the image transform.
    # Args: root: HAM10000 folder (images are in root/images/<image_id>.jpg).
    #       df: metadata rows of this split.
    #       transform: steps applied to each image (see transforms.py), or None.
    # Returns: nothing (it builds the object).
    def __init__(self, root: str, df: pd.DataFrame, transform=None):
        """Sets up the HAM10000 dataset for one split (train, val or test).
        It saves the image folder as a Path.
        It resets the row numbers of the table so they run from 0 to n-1.
        It also saves the transform that will be applied to each image.
        Input: root folder, the table rows of this split, and an optional transform.
        Output: nothing, it only prepares the object so images can be loaded later.
        """
        # remember the folder as a Path
        self.root = Path(root)
        # new row numbers 0..n-1 so iloc[idx] matches idx
        self.df = df.reset_index(drop=True)
        # remember the transform
        self.transform = transform

    # Class number of every image, as a numpy array.
    # Used to count images per class and to build the oversampling sampler.
    @property
    def labels(self):
        """Gives the class number of every image in this split.
        It takes the diagnosis (dx) column of the table.
        Each class name is changed into its number using CLASS_TO_IDX.
        The result is turned into a numpy array.
        Input: none, it only uses the stored table.
        Output: a numpy array of labels. It is needed to count classes and build the sampler.
        """
        # turn class names into numbers
        return self.df["dx"].map(CLASS_TO_IDX).to_numpy()

    # Returns how many images are in this split.
    def __len__(self):
        """Tells PyTorch how many images are in this split.
        It simply counts the rows of the stored table.
        Input: none, it uses the stored table only.
        Output: one whole number, the size of the split.
        The DataLoader needs this to know how many batches to make.
        It is a required method of every PyTorch Dataset.
        """
        # number of table rows
        return len(self.df)

    # Loads image number idx and returns it as a dict.
    # Args: idx: position from 0 to len-1 (the DataLoader calls this for you).
    # Returns: dict with image tensor, label, fitzpatrick (-1 here) and image_id.
    def __getitem__(self, idx):
        """Loads one image and its information by its position number.
        Step 1: take the table row at position idx.
        Step 2: open the jpg file and make sure it has 3 colour channels.
        Step 3: apply the transform (resize, augment, normalise) if there is one.
        Input: idx, a number from 0 to len-1.
        Output: a dict with image, label, fitzpatrick (always -1 here) and image_id.
        """
        # the metadata row for this image
        row = self.df.iloc[idx]
        # open the file; convert("RGB") makes sure it has 3 colour channels
        img = Image.open(self.root / "images" / f"{row.image_id}.jpg").convert("RGB")
        # apply augmentation and normalisation if a transform was given
        if self.transform:
            # replace the picture with the transformed tensor
            img = self.transform(img)
        # return the sample as a dict
        return {
            # the image (tensor)
            "image": img,
            # the class number of the diagnosis
            "label": CLASS_TO_IDX[row.dx],
            # HAM10000 has no skin-type information
            "fitzpatrick": -1,
            # the image's id
            "image_id": row.image_id,
        }


# Dataset of Fitzpatrick17k images, only for conditions that match the 7 HAM classes.
# Rows whose image file is missing are skipped. fitzpatrick is 1-6, or -1 if unknown.
# Why: this is the test set for the fairness study (accuracy per skin type).
class Fitzpatrick17kDataset(Dataset):
    # Reads the CSV and keeps rows that have a HAM class and a downloaded image.
    # Args: root: Fitzpatrick17k folder (images are in root/images/<md5hash>.jpg).
    #       transform: image steps or None.
    #       csv_name: name of the table file.
    # Returns: nothing (it builds the object).
    def __init__(self, root: str, transform=None, csv_name="fitzpatrick17k.csv"):
        """Sets up the Fitzpatrick17k dataset, used only for fairness testing.
        Step 1: read the CSV and map each condition name to a HAM class.
        Step 2: drop rows with no HAM class, then drop rows whose image file is missing.
        Step 3: rename columns and turn unknown skin types into -1.
        Input: root folder, an optional transform, and the CSV file name.
        Output: nothing, it prepares the object. It keeps only usable rows.
        """
        # remember the folder as a Path
        self.root = Path(root)
        # remember the transform
        self.transform = transform
        # read the table
        df = pd.read_csv(self.root / csv_name)
        # HAM class for each label (None if there is no match)
        df["dx"] = df["label"].map(map_label)
        # remember the row count to report later
        n_all = len(df)
        # drop rows with no HAM match
        df = df[df["dx"].notna()]
        # count of rows that have a match
        n_mapped = len(df)
        # for each row: is its image file on disk? (files are named by md5hash)
        has_img = df["md5hash"].map(lambda h: (self.root / "images" / f"{h}.jpg").exists())
        # keep only rows that have an image
        df = df[has_img]
        # warn if some images are missing
        if len(df) < n_mapped:
            # say how many are missing
            log.warning("Fitzpatrick17k: %d/%d mapped images missing on disk", n_mapped - len(df), n_mapped)
        # report how many rows were kept
        log.info("Fitzpatrick17k: kept %d of %d rows (mappable + downloaded)", len(df), n_all)
        # use HAM-style column names: image_id, and a short name (fst) for skin type
        df = df.rename(columns={"md5hash": "image_id", "fitzpatrick_scale": "fst"})
        # keep skin types 1-6; anything else (unknown or missing) becomes -1
        df["fst"] = df["fst"].where(df["fst"].between(1, 6), -1).astype(int)
        # new row numbers 0..n-1
        self.df = df.reset_index(drop=True)

    # Class number of every kept image, as a numpy array.
    @property
    def labels(self):
        """Gives the class number of every kept Fitzpatrick17k image.
        It takes the mapped diagnosis (dx) column of the table.
        Each class name is changed into its number using CLASS_TO_IDX.
        The result is turned into a numpy array.
        Input: none, it only uses the stored table.
        Output: a numpy array of labels, used when checking class counts.
        """
        # turn class names into numbers
        return self.df["dx"].map(CLASS_TO_IDX).to_numpy()

    # Returns how many images were kept.
    def __len__(self):
        """Tells PyTorch how many Fitzpatrick17k images were kept.
        It simply counts the rows of the stored table.
        Input: none, it uses the stored table only.
        Output: one whole number, the size of this dataset.
        The DataLoader needs this to know how many batches to make.
        It is a required method of every PyTorch Dataset.
        """
        # number of table rows
        return len(self.df)

    # Loads image number idx and returns it as a dict.
    # Args: idx: position from 0 to len-1.
    # Returns: dict with image tensor, label (0-6), fitzpatrick (skin type 1-6 or -1) and
    #          image_id. The fairness evaluation groups results by the skin type.
    def __getitem__(self, idx):
        """Loads one Fitzpatrick17k image and its information by position.
        Step 1: take the table row at position idx.
        Step 2: open the jpg file and make sure it has 3 colour channels.
        Step 3: apply the transform if there is one.
        Input: idx, a number from 0 to len-1.
        Output: a dict with image, label, fitzpatrick (skin type 1-6 or -1) and image_id.
        """
        # the table row for this image
        row = self.df.iloc[idx]
        # open the file as a 3-channel image
        img = Image.open(self.root / "images" / f"{row.image_id}.jpg").convert("RGB")
        # apply the transform if one was given
        if self.transform:
            # replace the picture with the transformed tensor
            img = self.transform(img)
        # return the sample as a dict
        return {
            # the image (tensor)
            "image": img,
            # the class number of the HAM-mapped diagnosis
            "label": CLASS_TO_IDX[row.dx],
            # skin type 1-6, or -1 if unknown
            "fitzpatrick": int(row.fst),
            # the image's id
            "image_id": row.image_id,
        }


# Builds the DataLoaders: train, val, test, and fitz (only if its CSV exists).
# Args: cfg: config dict; uses cfg["data"] (ham_root, fitz_root, image_size, batch_size,
#       num_workers, oversample) and cfg["seed"].
# Returns: dict {"train", "val", "test"[, "fitz"]} of DataLoaders. Each gives batches: a
#          dict where "image" is [batch, 3, size, size] and "label" is [batch].
# Why: one place that prepares all the data the rest of the project needs.
def get_dataloaders(cfg):
    """Builds all the DataLoaders the project needs from the config.
    Step 1: read the metadata and split it by lesion into train, val and test.
    Step 2: check there is no leakage, then make the datasets.
    Step 3: build the train loader (balanced sampler or shuffle) and val/test loaders.
    Step 4: add a fitz loader only if its CSV file exists.
    Input: cfg, the config dict. Output: a dict of DataLoaders, one place for all data.
    """
    # the data settings from the config
    d = cfg["data"]
    # seed for repeatable randomness (default 42)
    seed = cfg.get("seed", 42)
    # side length images are resized to
    size = d.get("image_size", 224)
    # worker processes for loading; 0 means the main process
    nw = d.get("num_workers", 0)

    # read the metadata table
    meta = load_ham_metadata(d["ham_root"])
    # split it by lesion into train, val and test tables
    train_df, val_df, test_df = lesion_level_split(meta, seed=seed)
    # stop if one lesion is in two splits (data leakage)
    assert_no_leakage(train_df, val_df, test_df)

    # training set: random augmentation
    train_ds = HAM10000Dataset(d["ham_root"], train_df, train_transforms(size))
    # validation set: fixed transform
    val_ds = HAM10000Dataset(d["ham_root"], val_df, eval_transforms(size))
    # test set: fixed transform
    test_ds = HAM10000Dataset(d["ham_root"], test_df, eval_transforms(size))

    # DataLoader settings shared by all loaders
    common = dict(
        # how many images per batch (default 32)
        batch_size=d.get("batch_size", 32),
        # how many worker processes
        num_workers=nw,
        # faster copy to the GPU when a GPU exists
        pin_memory=torch.cuda.is_available(),
        # keep workers alive between epochs (only allowed when there are workers)
        persistent_workers=nw > 0,
    )
    # if oversampling is turned on in the config
    if d.get("oversample", False):
        # balanced sampler picks rare classes more often (it replaces shuffle)
        sampler = make_balanced_sampler(train_ds.labels, num_classes=len(CLASSES), seed=seed)
        # training loader that uses the sampler
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
    else:
        # seeded generator so the shuffle order is repeatable
        g = torch.Generator().manual_seed(seed)
        # training loader with normal random shuffling
        train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)

    # collect the loaders; val and test are never shuffled so the order is always the same
    loaders = {
        # training loader built above
        "train": train_loader,
        # validation loader
        "val": DataLoader(val_ds, shuffle=False, **common),
        # test loader
        "test": DataLoader(test_ds, shuffle=False, **common),
    }

    # the Fitzpatrick folder is optional
    fitz_root = d.get("fitz_root")
    # only use it if its CSV file exists
    if fitz_root and (Path(fitz_root) / "fitzpatrick17k.csv").exists():
        # the extra test set
        fitz_ds = Fitzpatrick17kDataset(fitz_root, eval_transforms(size))
        # add its loader
        loaders["fitz"] = DataLoader(fitz_ds, shuffle=False, **common)
    else:
        # tell the user the fitz loader is left out
        log.warning("Fitzpatrick17k CSV not found under %s; 'fitz' loader omitted", fitz_root)
    # give back all the loaders
    return loaders
