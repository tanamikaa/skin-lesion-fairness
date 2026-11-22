"""Download Fitzpatrick17k images listed in the CSV (only the ones mappable to HAM classes).

Usage: python scripts/download_fitzpatrick.py --root data/fitzpatrick17k [--all] [--workers 8]

Owner: Afzal Raza. Expects `<root>/fitzpatrick17k.csv`
(https://github.com/mattgroh/fitzpatrick17k). Images are saved to `<root>/images/<md5hash>.jpg`.
Already-downloaded files are skipped, so it is safe to re-run. Some URLs are dead; failures are
printed and the rest still download.
"""
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from skinfair.data.fitzpatrick_mapping import map_label, mapping_report


def fetch(args):
    url, dest = args
    if dest.exists():
        return None
    try:
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

    rep = mapping_report(df)
    print(f"Mappable rows: {rep['kept']}  dropped: {rep['dropped']}")
    print("Per HAM class:", dict(rep["per_class"]))
    if rep["missing_keys"]:
        print("WARNING mapping keys not found in CSV (typo?):", rep["missing_keys"])

    if not args.all:
        df = df[df["label"].map(map_label).notna()]
    jobs = [(r.url, root / "images" / f"{r.md5hash}.jpg") for r in df.itertuples() if isinstance(r.url, str)]
    with ThreadPoolExecutor(args.workers) as ex:
        errors = [e for e in tqdm(ex.map(fetch, jobs), total=len(jobs)) if e]
    print(f"Done. {len(errors)} failed downloads.")
    for e in errors[:20]:
        print("  ", e)
