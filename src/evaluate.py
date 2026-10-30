"""Evaluation metrics, precision-recall analysis, and cost-benefit threshold tuning.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    scores: np.ndarray,
    label: str = "Evaluation",
    cost_fp: float = 10.0,
    cost_fn: float = 250.0,
) -> Dict[str, Any]:
    """Compute comprehensive classification and financial metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Binary predictions (0 or 1).
        scores: Continuous fraud risk scores / probabilities.
        label: Row or experiment identifier.
        cost_fp: Financial/operational cost of a False Alarm (investigation, customer friction).
        cost_fn: Financial cost of Missed Fraud (stolen funds, chargeback fees).

    Returns:
        Dict with metrics and financial loss calculation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total_financial_loss = (fp * cost_fp) + (fn * cost_fn)

    # Safe ROC-AUC calculation
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc = 0.0

    return {
        "Setting": label,
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC (AP)": round(pr_auc, 4),
        "Fraud Caught (TP)": tp,
        "False Alarms (FP)": fp,
        "Fraud Missed (FN)": fn,
        "Legit Cleared (TN)": tn,
        "Est. Financial Loss ($)": round(total_financial_loss, 2),
    }


def find_best_f1_threshold(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray,
) -> Tuple[float, float, pd.DataFrame]:
    """Find the threshold that maximizes F1 score on validation data.

    Returns:
        Tuple: (best_threshold, max_f1, search_dataframe)
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        return 0.5, 0.0, pd.DataFrame()

    denom = precision[:-1] + recall[:-1]
    f1_scores = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / (denom + 1e-12), 0.0)

    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    max_f1 = float(f1_scores[best_idx])

    table = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Precision": precision[:-1],
            "Recall": recall[:-1],
            "F1": f1_scores,
        }
    )

    return best_threshold, max_f1, table


def find_cost_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 250.0,
    candidate_thresholds: int = 100,
) -> Tuple[float, float, pd.DataFrame]:
    """Find the decision threshold that minimizes business loss.

    Formula: Loss = (FP * cost_fp) + (FN * cost_fn)

    Args:
        y_true: Ground truth binary labels.
        scores: Continuous fraud scores.
        cost_fp: Cost of customer friction / alert investigation.
        cost_fn: Cost of stolen money / chargeback.
        candidate_thresholds: Number of threshold steps to evaluate.

    Returns:
        Tuple: (optimal_threshold, min_loss, curve_dataframe)
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    thresholds = np.linspace(0.01, 0.99, candidate_thresholds)
    losses = []
    fps = []
    fns = []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp = int(((y_true == 0) & (preds == 1)).sum())
        fn = int(((y_true == 1) & (preds == 0)).sum())
        loss = (fp * cost_fp) + (fn * cost_fn)
        losses.append(loss)
        fps.append(fp)
        fns.append(fn)

    losses = np.array(losses)
    best_idx = int(np.argmin(losses))
    optimal_threshold = float(thresholds[best_idx])
    min_loss = float(losses[best_idx])

    curve_df = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Loss": losses,
            "False_Alarms": fps,
            "Missed_Fraud": fns,
        }
    )

    return optimal_threshold, min_loss, curve_df
