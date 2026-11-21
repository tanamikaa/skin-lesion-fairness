import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from skinfair.evaluation.fairness import (
    bootstrap_gap_ci,
    bootstrap_group_ci,
    compare_gap_bootstrap,
    fairness_gap,
    light_vs_dark,
    mcnemar_test,
    per_group_metrics,
)
from skinfair.evaluation.metrics import classification_metrics, fast_balanced_accuracy
from skinfair.evaluation.predict import PROB_COLS, predict


def make_df(n=600, acc_light=0.9, acc_dark=0.6, seed=0):
    rng = np.random.default_rng(seed)
    fst = rng.integers(1, 7, n)
    y = rng.integers(0, 7, n)
    ok = rng.random(n) < np.where(fst <= 3, acc_light, acc_dark)
    pred = np.where(ok, y, (y + rng.integers(1, 7, n)) % 7)
    return pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "label": y, "pred": pred, "fitzpatrick": fst})


def test_gap_zero_when_all_correct():
    df = pd.DataFrame({"label": [0, 1, 0, 1], "pred": [0, 1, 0, 1], "fitzpatrick": [1, 1, 5, 5]})
    assert fairness_gap(per_group_metrics(df))["gap"] == 0.0


def test_unknown_fst_ignored():
    df = pd.DataFrame({"label": [0, 1, 0], "pred": [0, 1, 1], "fitzpatrick": [-1, 2, 2]})
    assert per_group_metrics(df)["n"].tolist() == [2]


def test_fast_balanced_accuracy_matches_sklearn():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 7, 300), rng.integers(0, 7, 300)
    assert fast_balanced_accuracy(y, p) == pytest.approx(balanced_accuracy_score(y, p))


def test_classification_metrics_with_missing_classes_and_probs():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    prob = np.full((4, 7), 0.01)
    prob[np.arange(4), p] = 0.9
    m = classification_metrics(y, p, prob)
    assert m["accuracy"] == 0.75 and m["balanced_accuracy"] == pytest.approx(0.75)
    assert len(m["confusion_matrix"]) == 7 and m["per_class_recall"]["akiec"] == 0.5
    assert 0 <= m["macro_auroc"] <= 1


def test_light_vs_dark_and_bootstrap_ci_contains_gap_and_detects_difference():
    df = make_df(n=1500)
    ld = light_vs_dark(df).set_index("tone")
    assert ld.loc["light", "balanced_accuracy"] > ld.loc["dark", "balanced_accuracy"]
    r = bootstrap_gap_ci(df, n_boot=300)
    assert r["ci_low"] <= r["gap"] <= r["ci_high"] and r["ci_low"] > 0.1
    fair = bootstrap_gap_ci(make_df(1500, 0.8, 0.8, seed=2), n_boot=300)
    assert fair["ci_low"] < 0 < fair["ci_high"]


def test_bootstrap_requires_both_groups():
    df = make_df(100)
    with pytest.raises(ValueError):
        bootstrap_gap_ci(df[df["fitzpatrick"] <= 3])


def test_bootstrap_group_ci_columns():
    t = bootstrap_group_ci(make_df(), n_boot=50)
    assert t["fitzpatrick"].tolist() == [1, 2, 3, 4, 5, 6]
    assert (t["ci_low"] <= t["value"]).all() and (t["value"] <= t["ci_high"]).all()


def test_paired_gap_comparison_detects_fairer_model():
    a = make_df(2000, 0.85, 0.82, seed=3)          # fairer
    b = make_df(2000, 0.90, 0.55, seed=4)
    b["image_id"], b["label"], b["fitzpatrick"] = a["image_id"], a["label"], a["fitzpatrick"]
    rng = np.random.default_rng(5)  # regenerate b's preds against shared labels
    ok = rng.random(len(b)) < np.where(b["fitzpatrick"] <= 3, 0.9, 0.55)
    b["pred"] = np.where(ok, b["label"], (b["label"] + 1) % 7)
    r = compare_gap_bootstrap(a, b, n_boot=300)
    assert r["diff"] < 0 and r["p_value"] < 0.05 and r["ci_high"] < 0
    same = compare_gap_bootstrap(a, a, n_boot=100)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_mcnemar():
    a = pd.DataFrame({"image_id": list("abcdefgh"), "label": [0] * 8, "pred": [0] * 8})
    b = a.assign(pred=[1] * 8)
    r = mcnemar_test(a, b)
    assert r["a_only_correct"] == 8 and r["b_only_correct"] == 0 and r["p_value"] < 0.05
    assert mcnemar_test(a, a)["p_value"] == 1.0


def test_predict_table_matches_interface(tmp_path):
    import torch
    from skinfair.data.datasets import get_dataloaders
    from skinfair.models import build_model
    from .synthetic import make_cfg

    cfg = make_cfg(tmp_path)
    model = build_model(cfg).eval()
    df = predict(model, get_dataloaders(cfg)["fitz"], torch.device("cpu"))
    assert list(df.columns) == ["image_id", "label", "pred", *PROB_COLS, "fitzpatrick"]
    assert np.allclose(df[PROB_COLS].sum(1), 1, atol=1e-5)
    assert (df["pred"] == df[PROB_COLS].to_numpy().argmax(1)).all()
