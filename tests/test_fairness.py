# This file tests the fairness helpers.
# It checks that when every prediction is right, the gap between skin-type groups is 0.
# Run with: pytest tests/test_fairness.py

# pandas is used to make a small table
import pandas as pd

# the two functions we are testing
from skinfair.evaluation.fairness import fairness_gap, per_group_metrics


# Test: perfect predictions in every skin-type group should give a gap of exactly 0.
# Args: none. Returns: nothing (it fails if the assert is false).
def test_gap_zero_when_all_correct():
    """Tests that perfect predictions give a fairness gap of zero.
    Step 1: build a small table of 4 images where every prediction is right.
    Step 2: two images have skin type 1 and two have skin type 5.
    Step 3: compute the per-group scores, then the gap between best and worst group.
    Input: none. Output: nothing, the test fails if the gap is not 0.0.
    It is needed to make sure the fairness code is correct in the simplest case.
    """
    # 4 images where pred equals label in every row; skin type 1 has two images, type 5 has two
    df = pd.DataFrame({"label": [0, 1, 0, 1], "pred": [0, 1, 0, 1], "fitzpatrick": [1, 1, 5, 5]})
    # both groups score 1.0, so best minus worst must be 0.0
    assert fairness_gap(per_group_metrics(df))["gap"] == 0.0
