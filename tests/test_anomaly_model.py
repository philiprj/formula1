"""Tests for the anomaly/retirement prediction model."""

import numpy as np
import pandas as pd
import pytest

from f1deg.eval.metrics import compute_classification_metrics


@pytest.fixture
def anomaly_training_data():
    """Synthetic data with anomalous laps for model training."""
    np.random.seed(42)
    n = 500

    compounds = ["SOFT", "MEDIUM", "HARD"]
    drivers = ["VER", "HAM", "LEC", "NOR", "SAI"]
    circuits = ["Bahrain", "Jeddah", "Melbourne"]

    data = {
        "race_id": np.random.choice(["2024_01", "2024_02", "2024_03"], n),
        "circuit_id": np.random.choice(circuits, n),
        "driver_id": np.random.choice(drivers, n),
        "constructor_id": np.random.choice(
            ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston Martin"], n
        ),
        "lap_number": np.random.randint(2, 60, n),
        "tyre_life": np.random.randint(1, 30, n).astype(float),
        "compound": np.random.choice(compounds, n),
        "fuel_mass_kg": np.random.uniform(20, 110, n),
        "track_temp": np.random.normal(45, 5, n),
        "air_temp": np.random.normal(30, 3, n),
        "humidity": np.random.normal(40, 10, n),
        "wind_speed": np.random.normal(3, 1, n),
        "rainfall": np.random.choice([True, False], n, p=[0.05, 0.95]),
        "position": np.random.randint(1, 21, n).astype(float),
        "position_change": np.random.normal(0, 1, n),
        "traffic_density": np.random.randint(0, 5, n),
        "race_progress": np.random.uniform(0, 1, n),
        "stint_fraction": np.random.uniform(0, 1, n),
        "stint_number": np.random.choice([1, 2, 3], n),
        "stint_lap": np.random.randint(1, 25, n),
        "did_retire": [False] * n,
    }

    # Generate realistic lap times
    data["tyre_life_sq"] = data["tyre_life"] ** 2
    base = 90 - 0.035 * data["fuel_mass_kg"] + 0.05 * data["tyre_life"]
    data["lap_time_seconds"] = base + np.random.normal(0, 0.5, n)

    # Mark ~5% as anomalous
    anomalous_idx = np.random.choice(n, size=25, replace=False)
    data["is_anomalous_lap"] = np.zeros(n, dtype=bool)
    data["is_anomalous_lap"][anomalous_idx] = True
    # Make anomalous laps slower
    data["lap_time_seconds"][anomalous_idx] += np.random.uniform(5, 15, 25)

    return pd.DataFrame(data)


class TestAnomalyModel:
    def test_fit_and_predict(self, anomaly_training_data):
        """Model should fit and produce probabilities."""
        from f1deg.models.anomaly import AnomalyPredictionModel

        model = AnomalyPredictionModel()
        config = {
            "model": {
                "features": ["tyre_life", "fuel_mass_kg", "track_temp", "position"],
                "categorical_features": ["compound", "circuit_id"],
                "rolling_features": [],
                "rolling_window": 5,
                "target": "is_anomalous_lap",
            }
        }
        model.fit(anomaly_training_data, config)

        proba = model.predict_proba(anomaly_training_data)
        assert len(proba) == len(anomaly_training_data)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_binary(self, anomaly_training_data):
        """Binary predictions should be boolean array."""
        from f1deg.models.anomaly import AnomalyPredictionModel

        model = AnomalyPredictionModel()
        config = {
            "model": {
                "features": ["tyre_life", "fuel_mass_kg"],
                "categorical_features": ["compound"],
                "rolling_features": [],
                "rolling_window": 5,
                "target": "is_anomalous_lap",
            }
        }
        model.fit(anomaly_training_data, config)

        preds = model.predict(anomaly_training_data, threshold=0.5)
        assert preds.dtype == bool

    def test_save_and_load(self, anomaly_training_data, tmp_path):
        """Model should survive save/load round-trip."""
        from f1deg.models.anomaly import AnomalyPredictionModel

        model = AnomalyPredictionModel()
        config = {
            "model": {
                "features": ["tyre_life", "fuel_mass_kg"],
                "categorical_features": ["compound"],
                "rolling_features": [],
                "rolling_window": 5,
                "target": "is_anomalous_lap",
            }
        }
        model.fit(anomaly_training_data, config)
        proba_before = model.predict_proba(anomaly_training_data)

        model.save(tmp_path / "anomaly")
        loaded = AnomalyPredictionModel.load(tmp_path / "anomaly")
        proba_after = loaded.predict_proba(anomaly_training_data)

        np.testing.assert_array_almost_equal(proba_before, proba_after)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_classification_metrics(y_true, y_proba)
        assert metrics["auroc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100, p=[0.9, 0.1])
        y_proba = np.random.uniform(0, 1, 100)
        metrics = compute_classification_metrics(y_true, y_proba)
        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["brier_score"] <= 1

    def test_counts_correct(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = compute_classification_metrics(y_true, y_proba)
        assert metrics["positive_count"] == 2
        assert metrics["negative_count"] == 3
