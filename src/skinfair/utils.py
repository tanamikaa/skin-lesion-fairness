# This file holds small helpers used by several scripts.
# 1) apply_overrides changes config values using text like "train.lr=0.001".
# 2) run_name and run_dir decide the name and folder of a training run so runs do not overwrite each other.

# copy lets us duplicate a dict so we never change the caller's original.
import copy

# yaml turns text like "5" or "true" into real Python values.
import yaml


# apply_overrides: changes values in a config using command-line style text.
# Inputs: cfg (nested dict, e.g. {"train": {"lr": 0.01}}), overrides (list of strings like
#   ["train.lr=0.001", "seed=1"], or None).
# Returns: a new dict with the values replaced; the original cfg is not changed.
# Why: lets you try a setting without editing the YAML file.
def apply_overrides(cfg: dict, overrides) -> dict:
    """
    Changes values in a config using text such as train.lr=0.001.
    Steps: copy the config, split each item at the first equals sign.
    Then walk down the dotted key, creating levels if missing, and set the value.
    The value text is turned into a real value (5 becomes 5, true becomes True).
    Inputs: cfg (nested dict) and overrides (list of strings or None).
    Output: a new changed dict; the original is untouched, so you can try settings quickly.
    """
    # Make a full copy so the original config stays untouched.
    cfg = copy.deepcopy(cfg)
    # Loop over each override; "or []" means do nothing when overrides is None.
    for item in overrides or []:
        # Split the text at the first "=" into key, "=", and value text.
        key, _, raw = item.partition("=")
        # If there was no "=", the middle part is empty, so the input is wrong.
        if not _:
            # Stop with a helpful message.
            raise ValueError(f"Override must look like key=value, got {item!r}")
        # Start at the top of the config.
        node = cfg
        # "a.b.c" becomes parents ["a", "b"] and leaf "c".
        *parents, leaf = key.split(".")
        # Walk down through the parent levels.
        for k in parents:
            # Go one level down, creating the level if it is missing.
            node = node.setdefault(k, {})
        # Turn the value text into a real value ("5" -> 5, "true" -> True) and store it.
        node[leaf] = yaml.safe_load(raw)
    # Return the changed copy.
    return cfg


# run_name: builds a readable unique name for a run, like "resnet50_focal_os_s42".
# Input: cfg, the config dict.
# Returns: a string. If cfg["run_name"] is set it is used as is; otherwise the name is
#   <model folder>_<loss>[_os]_s<seed> ("os" = oversampling on).
# Why: the name is used for checkpoint and result files, so different runs need different names.
def run_name(cfg: dict) -> str:
    """
    Builds a readable and unique name for a training run.
    Steps: if cfg has run_name, return it as it is.
    Otherwise join the model folder, loss name, optional _os tag and seed.
    Input: cfg, the config dict.
    Output: a string such as resnet50_focal_os_s42.
    Checkpoint and result files use this name, so different runs must not clash.
    """
    # If the config gives a name, use it as is.
    if cfg.get("run_name"):
        # Return the given name.
        return cfg["run_name"]
    # Imported here because only this function needs it.
    from pathlib import Path

    # Last folder of output_dir, for example "resnet50".
    base = Path(cfg["output_dir"]).name
    # Add "_os" to the name when oversampling is on.
    os_tag = "_os" if cfg.get("data", {}).get("oversample") else ""
    # Join model, loss, oversample tag and seed so different runs never share a name.
    return f"{base}_{cfg['loss']['name']}{os_tag}_s{cfg.get('seed', 42)}"


# run_dir: gives the folder where a run saves its files.
# Input: cfg, the config dict.
# Returns: a Path like results/resnet50/resnet50_focal_s42 (the folder is not created here).
# Why: keeps all outputs of one run together in one place.
def run_dir(cfg: dict):
    """
    Gives the folder where one run saves its files.
    Steps: read output_dir from the config and add the run name.
    The folder is not created here; the caller does that.
    Input: cfg, the config dict.
    Output: a Path such as results/resnet50/resnet50_focal_s42.
    It keeps all outputs of one run together in one place.
    """
    # Imported here because only this function needs it.
    from pathlib import Path

    # Output folder plus run name (the caller creates the folder).
    return Path(cfg["output_dir"]) / run_name(cfg)
