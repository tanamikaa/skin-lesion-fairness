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
