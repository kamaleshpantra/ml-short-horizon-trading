from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def evaluate_classification(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | None = None,
    labels: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> dict[str, float | list[list[int]] | dict[str, float]]:
    """
    Compute comprehensive classification metrics for short-horizon market prediction.

    Parameters
    ----------
    y_true : np.ndarray | list
        Ground truth directional targets (-1, 0, 1).
    y_pred : np.ndarray | list
        Predicted directional targets (-1, 0, 1).
    y_prob : np.ndarray | None, default=None
        Predicted class probabilities (shape: [N, n_classes]).
    labels : tuple[float, ...], default=(-1.0, 0.0, 1.0)
        Expected target label values.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics: dict[str, float | list[list[int]] | dict[str, float]] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
        except Exception:
            metrics["log_loss"] = float("nan")

    return metrics
