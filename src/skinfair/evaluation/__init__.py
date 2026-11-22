"""Evaluation & fairness. Owner: Karanpal.

This package re-exports the main evaluation helpers so other code can write
`from skinfair.evaluation import predict, per_group_metrics, ...`.
"""
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
from .metrics import classification_metrics
from .predict import predict

# Names exported by `from skinfair.evaluation import *`.
__all__ = [
    "bootstrap_gap_ci", "bootstrap_group_ci", "classification_metrics", "compare_gap_bootstrap",
    "fairness_gap", "light_vs_dark", "mcnemar_test", "per_class_recall_light_vs_dark", "per_group_metrics", "predict",
]
