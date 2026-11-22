"""Classification metrics. Owner: Karanpal."""
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from skinfair import CLASSES, NUM_CLASSES

# Class indices 0..NUM_CLASSES-1; passed to sklearn so every class always gets a row/column.
LABELS = list(range(NUM_CLASSES))


def fast_balanced_accuracy(y_true, y_pred) -> float:
    """Mean per-class recall over classes present in y_true (== sklearn, but ~50x faster).

    Used inside bootstrap loops.

    Balanced accuracy = average of "fraction of class c images predicted correctly" over
    classes. Unlike plain accuracy it is not dominated by the most common class.
    Inputs: integer arrays of true and predicted labels. Returns a float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = max(int(y_true.max()), int(y_pred.max())) + 1  # number of bins needed for bincount
    support = np.bincount(y_true, minlength=n)  # how many images of each true class
    hits = np.bincount(y_true[y_true == y_pred], minlength=n)  # correct predictions per class
    present = support > 0  # only average over classes that actually appear in y_true
    return float((hits[present] / support[present]).mean())


def per_class_recall(y_true, y_pred):
    """Recall per class (list of 7). NaN, not 0, for classes absent from y_true (no support).

    Recall of class c = correctly predicted c images / all true c images.
    """
    # Confusion matrix: rows = true class, columns = predicted class.
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=LABELS)
    support = cm.sum(axis=1)  # row sums = number of true images per class
    with np.errstate(invalid="ignore", divide="ignore"):  # silence numpy warnings for 0/0
        # diagonal = correct predictions; max(support, 1) avoids division by zero,
        # and classes with no support get NaN.
        rec = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return [float(r) for r in rec]


def per_class_auroc(y_true, y_prob):
    """One-vs-rest AUROC per class; NaN for classes absent (or the only one) in y_true.

    For each class c, AUROC measures how well the probability for c ranks class-c images
    above all other images (1.0 = perfect, 0.5 = random).
    Inputs: true labels [N] and probabilities [N, 7]. Returns a list of 7 floats.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out = []
    for c in LABELS:
        pos = y_true == c  # boolean: is this image class c? (the "positive" class)
        # AUROC needs both positives and negatives, otherwise it is undefined.
        if pos.sum() == 0 or pos.sum() == len(pos):
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(pos, y_prob[:, c])))  # column c = P(class c)
    return out


def classification_metrics(y_true, y_pred, y_prob=None):
    """Overall metrics. y_prob: [N, 7] class probabilities (optional, enables AUROC).

    Classes absent from y_true get NaN recall / AUROC (not 0), so a subset such as one skin-type
    group is never reported as "0% recall" for a class it simply does not contain.
    `macro_auroc` averages one-vs-rest AUROC over the classes that actually occur in y_true,
    so subsets (e.g. one Fitzpatrick group) with missing classes do not crash.

    Inputs: true labels [N], predicted labels [N], optional probabilities [N, 7].
    Returns a dict with n, accuracy, balanced_accuracy, macro_f1, per_class_recall,
    per_class_support, confusion_matrix and, if y_prob is given, per_class_auroc and
    macro_auroc.
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
            "per_class_recall": dict(zip(CLASSES, per_class_recall(y_true, y_pred))),
            "per_class_support": dict(zip(CLASSES, np.bincount(y_true, minlength=NUM_CLASSES).tolist())),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        }
    if y_prob is not None:
        aucs = per_class_auroc(y_true, y_prob)
        out["per_class_auroc"] = dict(zip(CLASSES, aucs))
        valid = [a for a in aucs if not np.isnan(a)]  # skip classes with undefined AUROC
        out["macro_auroc"] = float(np.mean(valid)) if valid else float("nan")
    return out
