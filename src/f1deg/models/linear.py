"""Tier 1: Ridge regression baseline for tire degradation."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from f1deg.models.base import DegradationModel


class LinearDegradationModel(DegradationModel):
    """Ridge regression with one-hot encoded categoricals.

    Prediction intervals via residual standard error (assumes normality).
    """

    def __init__(self):
        self.model: Ridge | None = None
        self.encoder: OneHotEncoder | None = None
        self.residual_std: float = 0.0
        self.feature_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.config: dict = {}

    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        self.config = config
        model_config = config.get("model", {})
        alpha = model_config.get("alpha", 1.0)

        self.feature_cols = model_config.get(
            "features",
            ["tyre_life", "tyre_life_sq", "fuel_mass_kg", "track_temp", "air_temp", "rainfall"],
        )
        self.categorical_cols = model_config.get("categorical_features", ["compound", "circuit_id"])

        X, y = self._prepare_xy(train_df, fit_encoder=True)

        self.model = Ridge(alpha=alpha)
        self.model.fit(X, y)

        # Compute residual standard error for prediction intervals
        residuals = y - self.model.predict(X)
        self.residual_std = float(np.std(residuals))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        X, _ = self._prepare_xy(df, fit_encoder=False)
        return np.asarray(self.model.predict(X))

    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy import stats

        predictions = self.predict(df)
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * self.residual_std
        return predictions - margin, predictions + margin

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "encoder": self.encoder,
                    "residual_std": self.residual_std,
                    "feature_cols": self.feature_cols,
                    "categorical_cols": self.categorical_cols,
                    "config": self.config,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "LinearDegradationModel":
        instance = cls()
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        instance.model = data["model"]
        instance.encoder = data["encoder"]
        instance.residual_std = data["residual_std"]
        instance.feature_cols = data["feature_cols"]
        instance.categorical_cols = data["categorical_cols"]
        instance.config = data["config"]
        return instance

    def _prepare_xy(
        self, df: pd.DataFrame, fit_encoder: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepare feature matrix X and target y."""
        # Numeric features
        available_features = [c for c in self.feature_cols if c in df.columns]
        X_numeric = df[available_features].fillna(0).values.astype(float)

        # Convert boolean rainfall to numeric
        for i, col in enumerate(available_features):
            if col == "rainfall":
                X_numeric[:, i] = X_numeric[:, i].astype(float)

        # Categorical features
        available_cats = [c for c in self.categorical_cols if c in df.columns]
        if available_cats:
            cat_data = df[available_cats].fillna("UNKNOWN").astype(str)
            if fit_encoder:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                X_cat = self.encoder.fit_transform(cat_data)
            else:
                assert self.encoder is not None
                X_cat = self.encoder.transform(cat_data)
            X = np.hstack([X_numeric, X_cat])
        else:
            X = X_numeric

        y = df["lap_time_seconds"].values if "lap_time_seconds" in df.columns else None
        return X, y
