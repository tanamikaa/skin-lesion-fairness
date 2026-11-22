"""Small shared helpers."""
import copy

import yaml


def apply_overrides(cfg: dict, overrides) -> dict:
    """Return a copy of cfg with `a.b.c=value` overrides applied (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def run_name(cfg: dict) -> str:
    """Unique, readable name for a run, e.g. `resnet50_focal_os_s42`.

    Uses cfg["run_name"] if given; otherwise <base>_<loss>[_os]_s<seed> so that runs with a
    different loss / oversampling / seed never overwrite each other's logs or checkpoints.
    """
    if cfg.get("run_name"):
        return cfg["run_name"]
    from pathlib import Path

    base = Path(cfg["output_dir"]).name
    os_tag = "_os" if cfg.get("data", {}).get("oversample") else ""
    return f"{base}_{cfg['loss']['name']}{os_tag}_s{cfg.get('seed', 42)}"


def run_dir(cfg: dict):
    """Folder for this run's logs / predictions / metrics: <output_dir>/<run_name>."""
    from pathlib import Path

    return Path(cfg["output_dir"]) / run_name(cfg)
