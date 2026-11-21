import json

import pandas as pd
import torch

from skinfair.models import build_model, load_checkpoint
from skinfair.train import build_scheduler, train
from skinfair.utils import apply_overrides

from .synthetic import make_cfg


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = build_scheduler(opt, epochs=4, steps_per_epoch=10, warmup_epochs=1)
    lrs = []
    for _ in range(40):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        s.step()
    assert lrs[0] < lrs[9] <= 1.0 + 1e-9      # warm-up rises
    assert lrs[-1] < 0.05 and lrs[10] > lrs[30]  # cosine decays


def test_apply_overrides():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c.d=x", "e=[1,2]"])
    assert cfg == {"a": {"b": 2, "c": {"d": "x"}}, "e": [1, 2]}


def test_resnet_training_smoke(tmp_path):
    cfg = make_cfg(tmp_path, "resnet50", size=32, epochs=2)
    summary = train(cfg)
    log = pd.read_csv(summary["log_csv"])
    assert list(log["epoch"]) == [1, 2]
    assert log[["train_loss", "val_loss"]].notna().all().all()
    assert 0 <= summary["best_val_bal_acc"] <= 1
    assert json.load(open(tmp_path / "out" / "resnet50" / "summary.json"))["best_epoch"] >= 1
    m = build_model(cfg)
    ckpt = load_checkpoint(m, summary["checkpoint"])  # loadable
    assert ckpt["epoch"] == summary["best_epoch"]


def test_training_reduces_loss_on_tiny_problem(tmp_path):
    cfg = make_cfg(tmp_path, "resnet50", size=32, epochs=6)
    cfg["model"]["freeze_backbone_epochs"] = 0
    cfg["loss"]["name"] = "ce"
    cfg["train"]["mixup_alpha"] = 0.0
    train(cfg)
    log = pd.read_csv(tmp_path / "out" / "resnet50" / "log.csv")
    assert log["train_loss"].iloc[-1] < log["train_loss"].iloc[0]


def test_early_stopping(tmp_path, monkeypatch):
    import skinfair.train as T

    scores = iter([0.5, 0.4, 0.4, 0.4, 0.9, 0.9])  # best at epoch 1, then no improvement
    monkeypatch.setattr(T, "_validate", lambda *a, **k: (1.0, next(scores), 0.0))
    cfg = make_cfg(tmp_path, "resnet50", size=32, epochs=30)
    cfg["train"]["early_stopping_patience"] = 2
    summary = train(cfg)
    assert summary["epochs_run"] == 3 and summary["best_epoch"] == 1
