"""Evaluation metrics for tire degradation and anomaly prediction models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in seconds."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in seconds."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of true values falling within the prediction interval.

    Target: should match the nominal coverage (e.g., 0.95 for 95% intervals).
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def prediction_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals in seconds.

    Narrower is better, given calibrated coverage.
    """
    return float(np.mean(upper - lower))


def compound_ranking_accuracy(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    at_stint_lap: int = 15,
) -> float:
    """Fraction of races where predicted compound ranking matches actual.

    At a given stint_lap, rank compounds by predicted and actual lap time.
    A race is "correct" if the rankings match exactly.
    """
    df = df.copy()
    df["y_pred"] = y_pred

    if "stint_lap" not in df.columns or "compound" not in df.columns or "race_id" not in df.columns:
        return float("nan")

    # Filter to the target stint lap
    target = df[df["stint_lap"] == at_stint_lap]
    if target.empty:
        return float("nan")

    correct = 0
    total = 0

    for _race_id, race_group in target.groupby("race_id"):
        # Get mean actual and predicted per compound
        actual_ranking = (
            race_group.groupby("compound")["lap_time_seconds"].mean().sort_values().index.tolist()
        )
        pred_ranking = race_group.groupby("compound")["y_pred"].mean().sort_values().index.tolist()

        if len(actual_ranking) < 2:
            continue

        total += 1
        if actual_ranking == pred_ranking:
            correct += 1

    return correct / total if total > 0 else float("nan")


def auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Area Under ROC Curve for binary classification."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def auprc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Area Under Precision-Recall Curve.

    More informative than AUROC for imbalanced datasets.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score (lower is better). Measures calibration quality."""
    return float(brier_score_loss(y_true, y_proba))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all classification metrics for the anomaly model.

    Args:
        y_true: Binary ground truth (0/1).
        y_proba: Predicted probabilities.
        threshold: Decision threshold for precision/recall.

    Returns:
        Dict with metric names and values.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "auroc": auroc(y_true, y_proba),
        "auprc": auprc(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
    }


def mae_by_stint_phase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Compute MAE stratified by stint phase.

    Strategy-critical: errors in late-stint (cliff detection) and
    post-pit (fresh tire behavior) matter more than mid-stint accuracy.
    """
    if "stint_lap" not in df.columns:
        return {}

    stint_laps = df["stint_lap"].values
    results = {}

    # Early stint (post-pit fresh tire behavior, laps 1-5)
    early_mask = stint_laps <= 5
    if early_mask.sum() > 0:
        results["mae_stint_early"] = float(np.mean(np.abs(y_true[early_mask] - y_pred[early_mask])))

    # Mid stint (laps 6-19)
    mid_mask = (stint_laps > 5) & (stint_laps <= 19)
    if mid_mask.sum() > 0:
        results["mae_stint_mid"] = float(np.mean(np.abs(y_true[mid_mask] - y_pred[mid_mask])))

    # Late stint (lap 20+, cliff detection zone)
    late_mask = stint_laps >= 20
    if late_mask.sum() > 0:
        results["mae_stint_late"] = float(np.mean(np.abs(y_true[late_mask] - y_pred[late_mask])))

    return results


def mae_post_sc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Compute MAE for laps within 3 laps of SC restart.

    These are the most strategy-critical laps — when pit decisions
    have the highest impact.
    """
    if "laps_since_sc_end" not in df.columns:
        return {}

    sc_mask = df["laps_since_sc_end"].values <= 3
    if sc_mask.sum() == 0:
        return {}

    return {
        "mae_post_sc": float(np.mean(np.abs(y_true[sc_mask] - y_pred[sc_mask]))),
        "n_post_sc_laps": int(sc_mask.sum()),
    }


def degradation_trend_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
) -> float:
    """Fraction of consecutive laps where predicted degradation direction matches actual.

    For strategy, getting the *trend* right (times increasing = degradation) is
    as important as absolute accuracy. A model that correctly predicts "tires
    are getting worse" at the right time enables correct pit timing.
    """
    if (
        "race_id" not in df.columns
        or "driver_id" not in df.columns
        or "stint_lap" not in df.columns
    ):
        return float("nan")

    correct = 0
    total = 0

    temp = df.copy()
    temp["y_true"] = y_true
    temp["y_pred"] = y_pred

    for _, group in temp.groupby(
        ["race_id", "driver_id", "stint_number"]
        if "stint_number" in temp.columns
        else ["race_id", "driver_id"]
    ):
        sorted_g = group.sort_values("stint_lap")
        if len(sorted_g) < 2:
            continue

        true_delta = np.diff(sorted_g["y_true"].values)
        pred_delta = np.diff(sorted_g["y_pred"].values)

        # Direction match: both positive (getting slower) or both negative
        matches = np.sign(true_delta) == np.sign(pred_delta)
        correct += matches.sum()
        total += len(matches)

    return correct / total if total > 0 else float("nan")


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Returns dict with metric names and values.
    """
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

    if lower is not None and upper is not None:
        metrics["pi_coverage_95"] = prediction_interval_coverage(y_true, lower, upper)
        metrics["pi_width_mean"] = prediction_interval_width(lower, upper)

    if df is not None:
        for stint_lap in [10, 15, 25]:
            key = f"compound_ranking_acc_lap{stint_lap}"
            metrics[key] = compound_ranking_accuracy(df, y_pred, at_stint_lap=stint_lap)

        # Strategy-focused metrics
        metrics.update(mae_by_stint_phase(y_true, y_pred, df))
        metrics.update(mae_post_sc(y_true, y_pred, df))
        metrics["degradation_trend_accuracy"] = degradation_trend_accuracy(y_true, y_pred, df)

    return metrics
