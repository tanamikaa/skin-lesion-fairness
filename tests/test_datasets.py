import pandas as pd
from PIL import Image

from skinfair.data.datasets import CLASS_TO_IDX, Fitzpatrick17kDataset, HAM10000Dataset


def _make_image(path):
    Image.new("RGB", (8, 8), color=(100, 100, 100)).save(path)


def test_ham10000_dataset_item_shape(tmp_path):
    (tmp_path / "images").mkdir()
    for image_id in ("ISIC_0000001", "ISIC_0000002"):
        _make_image(tmp_path / "images" / f"{image_id}.jpg")

    df = pd.DataFrame(
        {
            "lesion_id": ["HAM_0001", "HAM_0002"],
            "image_id": ["ISIC_0000001", "ISIC_0000002"],
            "dx": ["mel", "nv"],
        }
    )
    ds = HAM10000Dataset(tmp_path, df)

    assert len(ds) == 2
    item = ds[0]
    assert item["label"] == CLASS_TO_IDX["mel"]
    assert item["fitzpatrick"] == -1
    assert item["image_id"] == "ISIC_0000001"
    assert item["image"].size == (8, 8)


def test_fitzpatrick17k_dataset_maps_and_drops_unmapped(tmp_path):
    (tmp_path / "images").mkdir()
    rows = [
        {"md5hash": "a1", "fitzpatrick_scale": 2, "label": "Melanoma"},
        {"md5hash": "a2", "fitzpatrick_scale": -1, "label": "nevocytic nevus"},
        {"md5hash": "a3", "fitzpatrick_scale": 4, "label": "some unmapped condition"},
    ]
    for row in rows:
        _make_image(tmp_path / "images" / f"{row['md5hash']}.jpg")
    pd.DataFrame(rows).to_csv(tmp_path / "fitzpatrick17k.csv", index=False)

    ds = Fitzpatrick17kDataset(tmp_path)

    assert len(ds) == 2  # the unmapped row is dropped
    labels = {ds[i]["image_id"]: ds[i] for i in range(len(ds))}
    assert labels["a1"]["label"] == CLASS_TO_IDX["mel"]
    assert labels["a1"]["fitzpatrick"] == 2
    assert labels["a2"]["label"] == CLASS_TO_IDX["nv"]
    assert labels["a2"]["fitzpatrick"] == -1
