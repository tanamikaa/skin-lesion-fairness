# This file turns a model's predictions into numbers that say how good the model is.
# It has plain scores (accuracy), class-fair scores (balanced accuracy, macro-F1),
# per-class scores (recall, AUROC) and the confusion matrix.
# It is part of the "evaluate" step of the pipeline (data -> model -> train -> evaluate -> explain).
# It only works on arrays of true labels and predictions, so any model can use it.
# fairness.py, scripts/evaluate.py and src/skinfair/train.py call the functions in here.
#
# Short word list:
#   - recall of one class: of the images that really are that class, the share the model found.
#   - balanced accuracy: the average recall of every class, so rare classes count as much as common ones.
#   - NaN: "not a number", used to mean "cannot be computed" (different from 0).

# warnings lets us hide a noisy sklearn warning later.
import warnings

# numpy is used for fast array maths.
import numpy as np
# These are ready-made scores from scikit-learn.
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Class names and the number of classes, shared by the whole project.
from skinfair import CLASSES, NUM_CLASSES

# A list of all class numbers 0..6, so every class always gets a row and a column.
LABELS = list(range(NUM_CLASSES))


# fast_balanced_accuracy: computes balanced accuracy (average recall per class).
# Inputs:
#   y_true: real classes of N images (1-D integers).
#   y_pred: predicted classes of N images (1-D integers).
# Returns: one float between 0 and 1 (only classes present in y_true are averaged).
# Why it exists: the bootstrap in fairness.py calls this thousands of times, and
# sklearn's version is slow, so this one uses only simple numpy counting.
def fast_balanced_accuracy(y_true, y_pred) -> float:
    """Compute balanced accuracy, which is the average recall of each class.
    Steps: turn inputs into arrays, count images per true class,
    count correct predictions per class, then divide hits by support.
    Only classes that really appear in y_true are averaged.
    Inputs: y_true and y_pred, two 1-D integer arrays of equal length.
    Returns one float between 0 and 1.
    Needed because rare classes must count as much as common ones.
    """
    # Turn the true labels into a numpy array.
    y_true = np.asarray(y_true)
    # Turn the predicted labels into a numpy array.
    y_pred = np.asarray(y_pred)
    # Number of bins needed so bincount covers the biggest class number.
    n = max(int(y_true.max()), int(y_pred.max())) + 1
    # Count how many images each true class has.
    support = np.bincount(y_true, minlength=n)
    # Count how many images of each class were predicted correctly.
    hits = np.bincount(y_true[y_true == y_pred], minlength=n)
    # True only for classes that appear in y_true (avoids dividing by zero).
    present = support > 0
    # Recall per present class, then the average of those recalls, as a plain float.
    return float((hits[present] / support[present]).mean())


# per_class_recall: computes recall for each of the 7 classes.
# Inputs:
#   y_true, y_pred: true and predicted classes (1-D integers, same length N).
# Returns: a list of 7 floats in class order; a class with no true images gives NaN, not 0.
# Why it exists: saying "0% recall" for an absent class would wrongly blame the model.
def per_class_recall(y_true, y_pred):
    """Find the recall of each of the 7 classes.
    Steps: build a confusion matrix, count images of each true class,
    then divide the correct ones (the diagonal) by that count.
    A class with no images gets NaN, meaning it cannot be computed.
    Inputs: y_true and y_pred, 1-D integer arrays.
    Returns a list of 7 floats (some may be NaN).
    Needed to see which classes the model misses.
    """
    # Confusion matrix: rows are true classes, columns are predicted classes.
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=LABELS)
    # Row sums give how many true images each class has.
    support = cm.sum(axis=1)
    # Hide numpy warnings for 0/0 because we handle that case ourselves.
    with np.errstate(invalid="ignore", divide="ignore"):
        # Correct predictions (the diagonal) divided by class size; NaN if the class is empty.
        rec = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    # Return plain Python floats, one per class.
    return [float(r) for r in rec]


# per_class_auroc: computes AUROC for each class (one class against all the others).
# AUROC answers: how often does the model give a higher probability to a random image of
# this class than to a random image of another class? 1.0 = always, 0.5 = coin flip.
# Inputs:
#   y_true: real classes, shape [N].
#   y_prob: probabilities, shape [N, 7] (one column per class).
# Returns: a list of 7 floats (NaN where the class is absent or is the only class).
def per_class_auroc(y_true, y_prob):
    """Compute a one-vs-rest AUROC score for each of the 7 classes.
    Steps: for each class, mark images of that class as positive and the rest negative,
    then score the class probability column with roc_auc_score.
    If a class has no positives or no negatives, store NaN instead.
    Inputs: y_true (N integers) and y_prob (N x 7 probabilities).
    Returns a list of 7 floats.
    Needed to judge how well each class is ranked, apart from any threshold.
    """
    # Turn the true labels into a numpy array.
    y_true = np.asarray(y_true)
    # Turn the probabilities into a numpy array.
    y_prob = np.asarray(y_prob)
    # This list will hold one AUROC per class.
    out = []
    # Go through each class one by one.
    for c in LABELS:
        # True where the image belongs to class c (a yes/no label for "class c vs rest").
        pos = y_true == c
        # AUROC needs both positive and negative images; if one kind is missing, store NaN.
        if pos.sum() == 0 or pos.sum() == len(pos):
            # Store NaN for this class.
            out.append(float("nan"))
        # Otherwise we can compute the score.
        else:
            # Use the probability column of class c as the score and store the AUROC.
            out.append(float(roc_auc_score(pos, y_prob[:, c])))
    # Give back the list of 7 values.
    return out


# classification_metrics: collects all the overall scores into one dict.
# Inputs:
#   y_true: real classes (1-D integers, length N).
#   y_pred: predicted classes (1-D integers, length N).
#   y_prob: optional probabilities [N, 7]; AUROC is only added if this is given.
# Returns: a dict with n, accuracy, balanced_accuracy, macro_f1, per_class_recall,
# per_class_support, confusion_matrix and (with y_prob) per_class_auroc and macro_auroc.
# Why it exists: HAM10000 has many more "nv" moles than other classes, so plain accuracy
# can look good while rare classes are missed; balanced accuracy and macro-F1 fix that.
def classification_metrics(y_true, y_pred, y_prob=None):
    """Collect all the overall scores into one dictionary.
    Steps: compute accuracy, balanced accuracy, macro-F1, per-class recall,
    per-class counts and the confusion matrix (sklearn warnings are hidden).
    If y_prob is given, also add per-class AUROC and the macro AUROC.
    Inputs: y_true, y_pred (1-D integers) and optional y_prob (N x 7).
    Returns a dict of these numbers.
    Needed because plain accuracy hides weak results on rare classes.
    """
    # Turn the true labels into a numpy array.
    y_true = np.asarray(y_true)
    # Turn the predicted labels into a numpy array.
    y_pred = np.asarray(y_pred)
    # Hide sklearn warnings about classes missing from small groups.
    with warnings.catch_warnings():
        # Ignore all warnings inside this block.
        warnings.simplefilter("ignore")
        # Build the dict of scores. Each key means:
        #   n: number of images.
        #   accuracy: share predicted correctly.
        #   balanced_accuracy: average recall over classes.
        #   macro_f1: F1 per class, averaged equally (zero_division=0 avoids errors for never-predicted classes).
        #   per_class_recall: recall for each named class.
        #   per_class_support: number of true images for each named class.
        #   confusion_matrix: nested list, rows = true, columns = predicted.
        out = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "per_class_recall": dict(zip(CLASSES, per_class_recall(y_true, y_pred))),
            "per_class_support": dict(zip(CLASSES, np.bincount(y_true, minlength=NUM_CLASSES).tolist())),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        }
    # Only compute AUROC when probabilities were given.
    if y_prob is not None:
        # AUROC for every class.
        aucs = per_class_auroc(y_true, y_prob)
        # Store them by class name.
        out["per_class_auroc"] = dict(zip(CLASSES, aucs))
        # Keep only the values that are not NaN so they do not break the average.
        valid = [a for a in aucs if not np.isnan(a)]
        # Average AUROC over valid classes, or NaN if there are none.
        out["macro_auroc"] = float(np.mean(valid)) if valid else float("nan")
    # Return all the scores.
    return out
