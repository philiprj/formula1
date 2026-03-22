"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from f1deg.eval.metrics import (
    compound_ranking_accuracy,
    compute_all_metrics,
    mae,
    prediction_interval_coverage,
    prediction_interval_width,
    rmse,
)


def test_mae():
    y_true = np.array([90, 91, 92])
    y_pred = np.array([90, 90, 90])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_mae_perfect():
    y = np.array([90, 91, 92])
    assert mae(y, y) == 0.0


def test_rmse():
    y_true = np.array([90, 91, 92])
    y_pred = np.array([90, 90, 90])
    # RMSE = sqrt((0 + 1 + 4) / 3) = sqrt(5/3) ≈ 1.291
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3), abs=1e-6)


def test_prediction_interval_coverage():
    y_true = np.array([90, 91, 92, 93, 94])
    lower = np.array([89, 90, 91, 91, 93])
    upper = np.array([91, 92, 93, 94, 95])
    # All 5 are within bounds
    assert prediction_interval_coverage(y_true, lower, upper) == 1.0


def test_prediction_interval_coverage_partial():
    y_true = np.array([90, 100])  # 100 is outside [89, 91]
    lower = np.array([89, 89])
    upper = np.array([91, 91])
    assert prediction_interval_coverage(y_true, lower, upper) == 0.5


def test_prediction_interval_width():
    lower = np.array([88, 89, 90])
    upper = np.array([92, 93, 94])
    assert prediction_interval_width(lower, upper) == 4.0


def test_compound_ranking_accuracy():
    df = pd.DataFrame(
        {
            "race_id": ["r1"] * 6,
            "compound": ["SOFT", "SOFT", "MEDIUM", "MEDIUM", "HARD", "HARD"],
            "stint_lap": [15, 15, 15, 15, 15, 15],
            "lap_time_seconds": [89, 90, 91, 92, 93, 94],  # SOFT < MEDIUM < HARD
        }
    )
    y_pred = np.array([89, 90, 91, 92, 93, 94])  # Same ranking
    assert compound_ranking_accuracy(df, y_pred, at_stint_lap=15) == 1.0


def test_compute_all_metrics():
    y_true = np.array([90.0, 91.0, 92.0])
    y_pred = np.array([90.5, 91.5, 91.5])
    lower = np.array([89.0, 90.0, 90.5])
    upper = np.array([92.0, 93.0, 93.0])

    metrics = compute_all_metrics(y_true, y_pred, lower, upper)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "pi_coverage_95" in metrics
    assert metrics["mae"] > 0
