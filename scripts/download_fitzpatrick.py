"""Download Fitzpatrick17k images listed in the CSV (only the ones mappable to HAM classes).

Usage: python scripts/download_fitzpatrick.py --root data/fitzpatrick17k [--all] [--workers 8]

Owner: Afzal Raza. Expects `<root>/fitzpatrick17k.csv`
(https://github.com/mattgroh/fitzpatrick17k). Images are saved to `<root>/images/<md5hash>.jpg`.
Already-downloaded files are skipped, so it is safe to re-run. Some URLs are dead; failures are
printed and the rest still download.

Arguments:
  --root     dataset folder holding fitzpatrick17k.csv (default data/fitzpatrick17k)
  --all      download every image, not only the ones mappable to HAM10000 classes
  --workers  number of parallel download threads (default 8)
"""
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from skinfair.data.fitzpatrick_mapping import map_label, mapping_report


def fetch(args):
    """Download one image.

    Input: args - a (url, dest) tuple; dest is the Path to save the file to.
    Returns: None on success or if the file already exists, else an error string "url: error".
    """
    url, dest = args  # one tuple, because ThreadPoolExecutor.map passes a single argument
    if dest.exists():
        return None  # already downloaded, skip
    try:
        # A browser-like User-Agent, since some servers block Python's default one.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return None
    except Exception as e:  # noqa: BLE001 - report and continue
        return f"{url}: {e}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/fitzpatrick17k")
    p.add_argument("--all", action="store_true", help="download all 114 conditions, not just mappable ones")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root = Path(args.root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "fitzpatrick17k.csv")

    # Summary of how CSV labels map to HAM classes: kept/dropped counts, per class, missing keys.
    rep = mapping_report(df)
    print(f"Mappable rows: {rep['kept']}  dropped: {rep['dropped']}")
    print("Per HAM class:", dict(rep["per_class"]))
    if rep["missing_keys"]:
        print("WARNING mapping keys not found in CSV (typo?):", rep["missing_keys"])

    if not args.all:
        # Keep only rows whose label maps to a HAM class (map_label gives None/NaN otherwise).
        df = df[df["label"].map(map_label).notna()]
    # One (url, save path) job per row; rows with a missing URL (NaN, not a str) are skipped.
    jobs = [(r.url, root / "images" / f"{r.md5hash}.jpg") for r in df.itertuples() if isinstance(r.url, str)]
    # Download in parallel threads; tqdm shows a progress bar; keep only the error strings.
    with ThreadPoolExecutor(args.workers) as ex:
        errors = [e for e in tqdm(ex.map(fetch, jobs), total=len(jobs)) if e]
    print(f"Done. {len(errors)} failed downloads.")
    for e in errors[:20]:  # show at most the first 20 errors
        print("  ", e)
