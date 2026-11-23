# Data (not tracked by git)

Owner: **Afzal Raza**

Nothing in this folder is committed. Team members can use the prepared copy on the team Drive (`skinfair_data/`),
or download the data themselves as described below.

Expected layout after download:
```
data/
  HAM10000/
    HAM10000_metadata.csv
    images/            # all 10,015 ISIC_*.jpg from part_1 + part_2
  fitzpatrick17k/
    fitzpatrick17k.csv
    images/            # <md5hash>.jpg for the 1,916 images that map onto the 7 HAM classes
```

## HAM10000
Official source, Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T
Download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` and `HAM10000_metadata`.

Alternative: the Kaggle mirror `kmader/skin-cancer-mnist-ham10000` (same images and metadata). On Colab or Linux,
after uploading your `kaggle.json` API token:
```bash
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
mkdir -p data/HAM10000/images
find data/HAM10000 -path data/HAM10000/images -prune -o -iname "*.jpg" -exec mv -t data/HAM10000/images {} +
ls data/HAM10000/images | wc -l    # should be 10015
```

## Fitzpatrick17k
CSV and image URLs: https://github.com/mattgroh/fitzpatrick17k (`fitzpatrick_scale` is 1-6, -1 = unknown).
Put `fitzpatrick17k.csv` in `data/fitzpatrick17k/`. Licence: CC BY-NC-SA 3.0.

1. Try the original URLs for the mappable conditions (re-runnable, skips existing files):
   ```bash
   python scripts/download_fitzpatrick.py --root data/fitzpatrick17k
   ```
   Many links are now dead: only 400 of the 1,916 mappable images downloaded this way.
2. The other 1,516 images come from an unofficial Kaggle mirror of the original, unprocessed images,
   already named by their MD5 hash. Do **not** use `nazmusresan/fitzpatrick17k`: its images have the background
   removed, which would affect a skin-tone study.
   ```bash
   kaggle datasets download -d mobaswiralfarabi/fitzpatrick17k-original -p data/fitzpatrick17k/raw --unzip
   find data/fitzpatrick17k/raw -iname "*.jpg" -exec mv -n -t data/fitzpatrick17k/images {} +
   ```
3. Check how many mapped images were found (should be 1,916):
   ```bash
   python -c "import logging; logging.basicConfig(level=logging.INFO); from skinfair.data.datasets import Fitzpatrick17kDataset; print(len(Fitzpatrick17kDataset('data/fitzpatrick17k')))"
   ```

The official alternative for missing images is the authors' request form, linked from the Fitzpatrick17k repository.
Label mapping to HAM classes lives in `src/skinfair/data/fitzpatrick_mapping.py`.
