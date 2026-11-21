"""Classification metrics. Owner: Karanpal."""
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from skinfair import CLASSES, NUM_CLASSES

LABELS = list(range(NUM_CLASSES))


def fast_balanced_accuracy(y_true, y_pred) -> float:
    """Mean per-class recall over classes present in y_true (== sklearn, but ~50x faster).

    Used inside bootstrap loops.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = max(int(y_true.max()), int(y_pred.max())) + 1
    support = np.bincount(y_true, minlength=n)
    hits = np.bincount(y_true[y_true == y_pred], minlength=n)
    present = support > 0
    return float((hits[present] / support[present]).mean())


def per_class_auroc(y_true, y_prob):
    """One-vs-rest AUROC per class; NaN for classes absent (or the only one) in y_true."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out = []
    for c in LABELS:
        pos = y_true == c
        if pos.sum() == 0 or pos.sum() == len(pos):
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(pos, y_prob[:, c])))
    return out


def classification_metrics(y_true, y_pred, y_prob=None):
    """Overall metrics. y_prob: [N, 7] class probabilities (optional, enables AUROC).

    `macro_auroc` averages one-vs-rest AUROC over the classes that actually occur in y_true,
    so subsets (e.g. one Fitzpatrick group) with missing classes do not crash.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns when y_pred has classes absent from y_true
        out = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "per_class_recall": dict(zip(
                CLASSES,
                (float(r) for r in recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)),
            )),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        }
    if y_prob is not None:
        aucs = per_class_auroc(y_true, y_prob)
        out["per_class_auroc"] = dict(zip(CLASSES, aucs))
        valid = [a for a in aucs if not np.isnan(a)]
        out["macro_auroc"] = float(np.mean(valid)) if valid else float("nan")
    return out
