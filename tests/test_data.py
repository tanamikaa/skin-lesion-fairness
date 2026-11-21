import numpy as np
import pandas as pd
import pytest

from skinfair import CLASSES
from skinfair.data.datasets import Fitzpatrick17kDataset, HAM10000Dataset, get_dataloaders, load_ham_metadata
from skinfair.data.fitzpatrick_mapping import FITZ_TO_HAM, map_label, mapping_report
from skinfair.data.sampling import make_balanced_sampler
from skinfair.data.splits import assert_no_leakage, lesion_level_split
from skinfair.data.transforms import eval_transforms, train_transforms

from .synthetic import make_cfg, make_synthetic_fitz, make_synthetic_ham


def test_mapping_targets_are_valid_ham_classes():
    assert set(FITZ_TO_HAM.values()) <= set(CLASSES)
    assert set(FITZ_TO_HAM.values()) == set(CLASSES)  # every HAM class reachable
    assert map_label("  Melanoma ") == "mel"
    assert map_label("psoriasis") is None
    assert map_label("squamous cell carcinoma") is None  # deliberately unmapped


def test_mapping_report_counts():
    df = pd.DataFrame({"label": ["melanoma", "psoriasis", "pyogenic granuloma", "Melanoma"]})
    rep = mapping_report(df)
    assert rep["kept"] == 3 and rep["dropped"] == 1
    assert rep["per_class"]["mel"] == 2
    assert "dermatofibroma" in rep["missing_keys"]


def test_split_has_no_lesion_leakage(tmp_path):
    meta = load_ham_metadata(make_synthetic_ham(tmp_path))
    tr, va, te = lesion_level_split(meta, seed=1)
    assert_no_leakage(tr, va, te)
    assert len(tr) + len(va) + len(te) == len(meta)
    assert set(tr["dx"]) == set(va["dx"]) == set(te["dx"]) == set(CLASSES)  # stratified


def test_leakage_check_catches_overlap(tmp_path):
    meta = load_ham_metadata(make_synthetic_ham(tmp_path))
    with pytest.raises(AssertionError):
        assert_no_leakage(meta, meta.iloc[:5], meta.iloc[5:10])


def test_split_is_deterministic(tmp_path):
    meta = load_ham_metadata(make_synthetic_ham(tmp_path))
    a = lesion_level_split(meta, seed=3)[2]["image_id"].tolist()
    b = lesion_level_split(meta, seed=3)[2]["image_id"].tolist()
    assert a == b


def test_balanced_sampler_balances_classes():
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    idx = np.fromiter(make_balanced_sampler(labels, seed=0), dtype=int)
    frac = np.bincount(labels[idx], minlength=3) / len(idx)
    assert np.allclose(frac, 1 / 3, atol=0.05)


def test_transform_shapes():
    from PIL import Image
    img = Image.new("RGB", (100, 80))
    assert train_transforms(64)(img).shape == (3, 64, 64)
    assert eval_transforms(64)(img).shape == (3, 64, 64)


def test_fitzpatrick_dataset_filters_and_labels(tmp_path):
    root = make_synthetic_fitz(tmp_path)
    ds = Fitzpatrick17kDataset(root, eval_transforms(32))
    assert len(ds) == 7 * 12  # 'psoriasis' dropped
    item = ds[0]
    assert set(item) == {"image", "label", "fitzpatrick", "image_id"}
    assert 1 <= item["fitzpatrick"] <= 6
    assert ds.labels.min() >= 0 and ds.labels.max() <= 6


def test_ham_item_interface(tmp_path):
    root = make_synthetic_ham(tmp_path)
    ds = HAM10000Dataset(root, load_ham_metadata(root), eval_transforms(32))
    item = ds[0]
    assert item["image"].shape == (3, 32, 32) and item["fitzpatrick"] == -1


def test_get_dataloaders_batch_interface(tmp_path):
    loaders = get_dataloaders(make_cfg(tmp_path))
    assert set(loaders) == {"train", "val", "test", "fitz"}
    b = next(iter(loaders["train"]))
    assert b["image"].shape[1:] == (3, 32, 32)
    assert b["label"].dtype.is_floating_point is False and len(b["image_id"]) == len(b["label"])
    assert (b["fitzpatrick"] == -1).all()
    assert (next(iter(loaders["fitz"]))["fitzpatrick"] >= 1).all()


def test_fitz_loader_omitted_when_missing(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["data"]["fitz_root"] = str(tmp_path / "nope")
    assert "fitz" not in get_dataloaders(cfg)
