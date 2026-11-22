# ---------------------------------------------------------------------------
# download_fitzpatrick.py: downloads the Fitzpatrick17k images.
#
# Usage:
#   python scripts/download_fitzpatrick.py --root data/fitzpatrick17k [--all] [--workers 8]
#
# What it does: reads <root>/fitzpatrick17k.csv (a list of image URLs) and saves the
# images to <root>/images/<md5hash>.jpg. By default it downloads only images whose
# label can be mapped to a HAM10000 class; --all downloads everything.
# Files that already exist are skipped, so running it again is safe. Dead URLs are
# printed and the rest still download.
#
# Where it fits in the pipeline: it is the very first step (getting the data),
# before data loading, training and evaluation.
# ---------------------------------------------------------------------------

# Lets us read options typed on the command line.
import argparse
# Used to download files from the web.
import urllib.request
# Runs many downloads at the same time.
from concurrent.futures import ThreadPoolExecutor
# Path makes file and folder paths easy to handle.
from pathlib import Path

# pandas reads the CSV table.
import pandas as pd
# Shows a progress bar.
from tqdm import tqdm

# Helpers that match Fitzpatrick labels to HAM classes.
from skinfair.data.fitzpatrick_mapping import map_label, mapping_report


# fetch: downloads ONE image.
# Input: args, a pair (url, dest). url is the web address, dest is the Path to save to.
#   It is one tuple because the thread pool passes a single argument to the function.
# Returns: None if all went well (or the file was already there), else an error text.
# Why it exists: it is the small job that every download thread runs, and it must
#   never crash, so one bad link does not stop all the others.
def fetch(args):
    """
    This function downloads one image from the internet.
    Input: args, a pair (url, dest): the web address and the Path to save to.
    First it splits the pair, and skips the work if the file already exists.
    Then it sends a browser-like request and reads the bytes within 30 seconds.
    It writes those bytes to the dest file on disk.
    It returns None on success, or an error text if anything fails.
    It is needed so one bad link never stops the other download threads.
    """
    # Split the pair into the web address and the save path.
    url, dest = args
    # If the file is already saved, do nothing.
    if dest.exists():
        # None means "no error".
        return None
    # Try the download, but do not crash if it fails.
    try:
        # Build a request that pretends to be a browser, because some sites block Python.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        # Open the URL and give up after 30 seconds.
        with urllib.request.urlopen(req, timeout=30) as r:
            # Read the downloaded bytes and save them to the file.
            dest.write_bytes(r.read())
        # Success, so no error text.
        return None
    # Catch any problem so one bad link does not stop the rest.
    except Exception as e:  # noqa: BLE001 - report and continue
        # Give back the error text so it can be printed later.
        return f"{url}: {e}"


# Run the code below only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Create the object that reads command-line options.
    p = argparse.ArgumentParser()
    # Folder that holds the CSV and the images.
    p.add_argument("--root", default="data/fitzpatrick17k")
    # Flag: if given, download every image.
    p.add_argument("--all", action="store_true", help="download all 114 conditions, not just mappable ones")
    # How many downloads to run at the same time.
    p.add_argument("--workers", type=int, default=8)
    # Read the options the user typed and store them in args.
    args = p.parse_args()

    # Turn the folder text into a Path object.
    root = Path(args.root)
    # Create the images folder if it does not exist yet.
    (root / "images").mkdir(parents=True, exist_ok=True)
    # Load the CSV list of images.
    df = pd.read_csv(root / "fitzpatrick17k.csv")

    # Count how many rows map to HAM classes and how many do not.
    rep = mapping_report(df)
    # Print the kept and dropped counts.
    print(f"Mappable rows: {rep['kept']}  dropped: {rep['dropped']}")
    # Print the count for each HAM class.
    print("Per HAM class:", dict(rep["per_class"]))
    # Warn if some mapping names were not found in the CSV (maybe a typo).
    if rep["missing_keys"]:
        # Print the warning with the missing names.
        print("WARNING mapping keys not found in CSV (typo?):", rep["missing_keys"])

    # Unless --all was given, filter the rows.
    if not args.all:
        # Keep only rows whose label maps to a HAM class (other labels give None/NaN).
        df = df[df["label"].map(map_label).notna()]
    # Make a (url, save path) job for each row; skip rows with no URL (a missing URL is NaN, not a string).
    jobs = [(r.url, root / "images" / f"{r.md5hash}.jpg") for r in df.itertuples() if isinstance(r.url, str)]
    # Start a pool of download threads.
    with ThreadPoolExecutor(args.workers) as ex:
        # Run all jobs in parallel with a progress bar; keep only the non-empty error texts.
        errors = [e for e in tqdm(ex.map(fetch, jobs), total=len(jobs)) if e]
    # Tell how many downloads failed.
    print(f"Done. {len(errors)} failed downloads.")
    # Show only the first 20 errors to keep the output short.
    for e in errors[:20]:
        # Print one error, indented.
        print("  ", e)
