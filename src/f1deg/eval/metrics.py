"""Evaluation metrics for tire degradation models."""

import numpy as np
import pandas as pd


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

    return metrics
