import pandas as pd

from skinfair.evaluation.fairness import fairness_gap, per_group_metrics


def test_gap_zero_when_all_correct():
    df = pd.DataFrame({"label": [0, 1, 0, 1], "pred": [0, 1, 0, 1], "fitzpatrick": [1, 1, 5, 5]})
    assert fairness_gap(per_group_metrics(df))["gap"] == 0.0
