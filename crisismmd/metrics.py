"""Classification metrics computed over all classes of a task."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels, positive=None):
    """Accuracy, macro/weighted F1, per-class scores (incl. number correct) and confusion matrix.

    labels: class names in id order; all classes are included even if never predicted.
    positive: optional class name whose precision/recall is reported separately (Part A).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=ids,
                                                                     zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=ids)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i]),
                              "correct": int(matrix[i, i])}
                      for i, label in enumerate(labels)},
        "confusion_matrix": matrix.tolist(),
        "labels": list(labels),
    }
    if positive is not None:
        result[f"{positive}_precision"] = result["per_class"][positive]["precision"]
        result[f"{positive}_recall"] = result["per_class"][positive]["recall"]
    return result
