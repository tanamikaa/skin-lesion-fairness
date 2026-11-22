# This package holds everything used AFTER training to judge a model:
# predict.py (run the model), metrics.py (overall scores) and fairness.py (scores per skin
# tone, gaps, confidence intervals and model comparison). plots.py draws charts and is
# imported directly. It is the "evaluate" step of the pipeline.
# This __init__.py only re-exports names, so you can write
# `from skinfair.evaluation import predict` instead of the longer module path.

# Bring the fairness functions up to package level so they can be imported from one place.
from .fairness import (
    bootstrap_gap_ci,
    bootstrap_group_ci,
    compare_gap_bootstrap,
    fairness_gap,
    light_vs_dark,
    mcnemar_test,
    per_class_recall_light_vs_dark,
    per_group_metrics,
)
# Re-export the overall metrics function.
from .metrics import classification_metrics
# Re-export the function that runs a model and makes predictions.
from .predict import predict

# These are the names that `from skinfair.evaluation import *` will give.
__all__ = [
    "bootstrap_gap_ci", "bootstrap_group_ci", "classification_metrics", "compare_gap_bootstrap",
    "fairness_gap", "light_vs_dark", "mcnemar_test", "per_class_recall_light_vs_dark", "per_group_metrics", "predict",
]
