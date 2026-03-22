"""Anomaly/retirement prediction model.

Predicts the probability that a given lap is anomalous (outlier, incident,
or precursor to retirement). Uses LightGBM binary classification with
calibrated probabilities.
"""

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class AnomalyPredictionModel:
    """LightGBM binary classifier for anomalous lap prediction.

    Unlike DegradationModel (regression), this outputs calibrated
    probabilities of a lap being anomalous.
    """

    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.rolling_feature_cols: list[str] = []
        self.config: dict = {}

    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        """Train the anomaly classifier.

        Args:
            train_df: Full lap DataFrame with is_anomalous_lap target column.
            config: Model configuration dict.
        """
        import lightgbm as lgb

        self.config = config
        model_config = config.get("model", {})

        self.feature_cols = model_config.get(
            "features",
            [
                "tyre_life",
                "tyre_life_sq",
                "fuel_mass_kg",
                "track_temp",
                "air_temp",
                "humidity",
                "wind_speed",
                "rainfall",
                "position",
                "position_change",
                "traffic_density",
                "race_progress",
                "stint_fraction",
            ],
        )
        self.categorical_cols = model_config.get(
            "categorical_features",
            ["compound", "circuit_id", "driver_id", "constructor_id"],
        )
        self.rolling_feature_cols = model_config.get(
            "rolling_features",
            [
                "lap_time_delta",
                "lap_time_trend",
                "position_volatility",
                "tyre_age_vs_typical",
            ],
        )

        # Compute rolling features
        df = self._add_rolling_features(train_df)

        target_col = model_config.get("target", "is_anomalous_lap")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")

        X, y = self._prepare_xy(df, target_col=target_col, fit_encoders=True)

        lgb_params = model_config.get("lightgbm", {})
        params = {
            "n_estimators": lgb_params.get("n_estimators", 500),
            "max_depth": lgb_params.get("max_depth", 6),
            "learning_rate": lgb_params.get("learning_rate", 0.05),
            "subsample": lgb_params.get("subsample", 0.8),
            "colsample_bytree": lgb_params.get("colsample_bytree", 0.8),
            "min_child_samples": lgb_params.get("min_child_samples", 20),
            "reg_alpha": lgb_params.get("reg_alpha", 0.1),
            "reg_lambda": lgb_params.get("reg_lambda", 1.0),
            "is_unbalance": lgb_params.get("is_unbalance", True),
            "verbose": -1,
        }

        self.model = lgb.LGBMClassifier(objective="binary", **params)
        self.model.fit(X, y)

        # Calibrate probabilities
        calibration_method = model_config.get("calibration", "isotonic")
        self.calibrated_model = CalibratedClassifierCV(self.model, method=calibration_method, cv=3)
        self.calibrated_model.fit(X, y)

        assert y is not None, "Target column not found in DataFrame"
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        logger.info(
            f"Trained anomaly model: {len(y)} samples "
            f"({pos_count} positive, {neg_count} negative, "
            f"ratio={pos_count / max(len(y), 1):.3f})"
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of anomalous lap.

        Returns array of probabilities (0-1).
        """
        if self.calibrated_model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        df = self._add_rolling_features(df)
        X, _ = self._prepare_xy(df, fit_encoders=False)
        return self.calibrated_model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary anomalous/normal label.

        Args:
            df: Feature DataFrame.
            threshold: Probability threshold for positive class.

        Returns:
            Boolean array.
        """
        proba = self.predict_proba(df)
        return proba >= threshold

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the underlying LightGBM model."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        importances = self.model.feature_importances_
        names = self.model.feature_name_
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "anomaly_model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "calibrated_model": self.calibrated_model,
                    "label_encoders": self.label_encoders,
                    "feature_cols": self.feature_cols,
                    "categorical_cols": self.categorical_cols,
                    "rolling_feature_cols": self.rolling_feature_cols,
                    "config": self.config,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "AnomalyPredictionModel":
        """Deserialize model from disk."""
        instance = cls()
        with open(path / "anomaly_model.pkl", "rb") as f:
            data = pickle.load(f)
        instance.model = data["model"]
        instance.calibrated_model = data["calibrated_model"]
        instance.label_encoders = data["label_encoders"]
        instance.feature_cols = data["feature_cols"]
        instance.categorical_cols = data["categorical_cols"]
        instance.rolling_feature_cols = data["rolling_feature_cols"]
        instance.config = data["config"]
        return instance

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling/derived features for anomaly detection."""
        result = df.copy()
        window = self.config.get("model", {}).get("rolling_window", 5)

        driver_col = "driver_id" if "driver_id" in result.columns else "Driver"
        race_col = "race_id" if "race_id" in result.columns else None
        time_col = "lap_time_seconds"

        group_cols = [driver_col]
        if race_col and race_col in result.columns:
            group_cols = [race_col, driver_col]

        if time_col in result.columns:
            # Lap time delta: difference from rolling median
            rolling_median = result.groupby(group_cols)[time_col].transform(
                lambda x: x.rolling(window, min_periods=2).median()
            )
            result["lap_time_delta"] = result[time_col] - rolling_median

            # Lap time trend: rolling slope (approximated as diff of rolling mean)
            rolling_mean = result.groupby(group_cols)[time_col].transform(
                lambda x: x.rolling(window, min_periods=2).mean()
            )
            result["lap_time_trend"] = rolling_mean.groupby([result[c] for c in group_cols]).diff()

        # Position volatility
        if "position" in result.columns:
            result["position_volatility"] = result.groupby(group_cols)["position"].transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )

        # Tyre age vs typical: how far past median stint length for compound/circuit
        if "tyre_life" in result.columns and "compound" in result.columns:
            stint_group = ["compound"]
            if "circuit_id" in result.columns:
                stint_group.append("circuit_id")
            median_stint = result.groupby(stint_group)["tyre_life"].transform("median")
            result["tyre_age_vs_typical"] = result["tyre_life"] - median_stint

        # Historical reliability (fraction of races completed per driver)
        if "did_retire" in result.columns and race_col and driver_col in result.columns:
            driver_race = result.groupby([race_col, driver_col])["did_retire"].first()
            driver_retire_rate = driver_race.groupby(driver_col).mean()
            result["historical_driver_reliability"] = 1.0 - result[driver_col].map(
                driver_retire_rate
            ).fillna(0.0)
        if "did_retire" in result.columns and race_col and "constructor_id" in result.columns:
            team_race = result.groupby([race_col, "constructor_id"])["did_retire"].first()
            team_retire_rate = team_race.groupby("constructor_id").mean()
            result["historical_constructor_reliability"] = 1.0 - result["constructor_id"].map(
                team_retire_rate
            ).fillna(0.0)

        # Wet on slicks
        if "compound" in result.columns and "rainfall" in result.columns:
            dry_compounds = {"SOFT", "MEDIUM", "HARD"}
            rainfall_bool = pd.array(result["rainfall"].values, dtype="boolean").fillna(False)
            result["is_wet_on_slicks"] = result["compound"].isin(dry_compounds) & rainfall_bool

        return result

    def _prepare_xy(
        self,
        df: pd.DataFrame,
        target_col: str = "is_anomalous_lap",
        fit_encoders: bool = False,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Prepare feature matrix and target vector.

        Returns a DataFrame (not numpy array) so LightGBM and
        CalibratedClassifierCV see consistent feature names.
        """
        all_numeric = self.feature_cols + [c for c in self.rolling_feature_cols if c in df.columns]
        # Add computed features that may exist
        extra_features = [
            "historical_driver_reliability",
            "historical_constructor_reliability",
            "is_wet_on_slicks",
        ]
        all_numeric += [c for c in extra_features if c in df.columns]

        available = [c for c in all_numeric if c in df.columns]
        X_parts = {col: df[col].fillna(0).astype(float) for col in available}

        available_cats = [c for c in self.categorical_cols if c in df.columns]
        for col in available_cats:
            if fit_encoders:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna("UNKNOWN").astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    continue
                vals = df[col].fillna("UNKNOWN").astype(str)
                encoded = np.array([le.transform([v])[0] if v in le.classes_ else 0 for v in vals])
            X_parts[f"enc_{col}"] = encoded.astype(float)

        X = pd.DataFrame(X_parts, index=df.index)

        y = df[target_col].values.astype(int) if target_col in df.columns else None
        return X, y
