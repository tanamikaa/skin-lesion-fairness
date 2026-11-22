"""Small shared helpers."""
import copy

import yaml


def apply_overrides(cfg: dict, overrides) -> dict:
    """Return a copy of cfg with `a.b.c=value` overrides applied (values parsed as YAML).

    Inputs: cfg = nested config dict; overrides = list of strings like "train.lr=0.001"
    (or None). Returns a new dict; the original cfg is not modified.
    Example: "train.epochs=5" sets cfg["train"]["epochs"] = 5 (an int, thanks to YAML parsing).
    """
    cfg = copy.deepcopy(cfg)  # deep copy so nested dicts of the caller are not changed
    for item in overrides or []:  # `or []` lets overrides be None
        key, _, raw = item.partition("=")  # split at the first "=": key, "=", value text
        if not _:  # partition returns "" as separator when there is no "=" at all
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")  # "a.b.c" -> parents ["a", "b"], leaf "c"
        for k in parents:
            node = node.setdefault(k, {})  # walk down, creating missing sub-dicts
        node[leaf] = yaml.safe_load(raw)  # "5" -> 5, "true" -> True, "[1,2]" -> list
    return cfg


def run_name(cfg: dict) -> str:
    """Unique, readable name for a run, e.g. `resnet50_focal_os_s42`.

    Uses cfg["run_name"] if given; otherwise <base>_<loss>[_os]_s<seed> so that runs with a
    different loss / oversampling / seed never overwrite each other's logs or checkpoints.

    Input: config dict (reads run_name, output_dir, loss.name, data.oversample, seed).
    Returns the run name as a string.
    """
    if cfg.get("run_name"):  # an explicit name in the config wins
        return cfg["run_name"]
    from pathlib import Path

    base = Path(cfg["output_dir"]).name  # last folder of output_dir, e.g. "resnet50"
    os_tag = "_os" if cfg.get("data", {}).get("oversample") else ""  # mark oversampled runs
    # e.g. resnet50 + focal + _os + _s42 -> "resnet50_focal_os_s42" (seed defaults to 42)
    return f"{base}_{cfg['loss']['name']}{os_tag}_s{cfg.get('seed', 42)}"


def run_dir(cfg: dict):
    """Folder for this run's logs / predictions / metrics: <output_dir>/<run_name>.

    Input: config dict. Returns a pathlib.Path (the folder is not created here).
    """
    from pathlib import Path

    return Path(cfg["output_dir"]) / run_name(cfg)
