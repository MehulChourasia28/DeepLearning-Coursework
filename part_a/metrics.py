"""Metrics required by Part A (computed over both classes; scale 0-1)."""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from . import config as C


def compute(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    out = {
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        # rows = true [informative, not_informative], cols = predicted
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    for i, lab in enumerate(C.LABELS):
        out[f"{lab}_precision"], out[f"{lab}_recall"], out[f"{lab}_f1"] = p[i], r[i], f[i]
        out[f"{lab}_support"] = int(s[i])
    return {k: (float(v) if isinstance(v, float) or hasattr(v, "item") else v)
            for k, v in out.items()}
