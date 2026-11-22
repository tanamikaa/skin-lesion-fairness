# Data (not tracked by git)

Owner: **Afzal Raza**

Expected layout after download:
```
data/
  HAM10000/
    HAM10000_metadata.csv
    images/            # all ISIC_*.jpg from part_1 + part_2
  fitzpatrick17k/
    fitzpatrick17k.csv
    images/            # downloaded from the URLs in the CSV
```

## HAM10000
Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T
Download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`, `HAM10000_metadata`.

## Fitzpatrick17k
https://github.com/mattgroh/fitzpatrick17k — CSV has image URLs + `fitzpatrick_scale` (1–6, −1 = unknown).
Download the images for the mappable conditions with:
```bash
python scripts/download_fitzpatrick.py --root data/fitzpatrick17k
```
(prints how many rows map to each HAM class and warns about mapping keys not found in the CSV; re-runnable, skips existing files).
Label mapping to HAM classes lives in `src/skinfair/data/fitzpatrick_mapping.py`.
