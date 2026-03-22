"""Tier 3: Gradient boosted trees (XGBoost/LightGBM) for tire degradation."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from f1deg.models.base import DegradationModel


class GBMDegradationModel(DegradationModel):
    """XGBoost or LightGBM with quantile regression for prediction intervals."""

    def __init__(self):
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.config: dict = {}
        self.backend: str = "xgboost"

    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        self.config = config
        model_config = config.get("model", {})
        self.backend = model_config.get("backend", "xgboost")

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
            ],
        )
        self.categorical_cols = model_config.get(
            "categorical_features",
            [
                "compound",
                "circuit_id",
                "driver_id",
                "constructor_id",
            ],
        )

        quantiles = model_config.get("quantiles", [0.025, 0.5, 0.975])

        X, y = self._prepare_xy(train_df, fit_encoders=True)

        if self.backend == "xgboost":
            self._fit_xgboost(X, y, model_config.get("xgboost", {}), quantiles)
        else:
            self._fit_lightgbm(X, y, model_config.get("lightgbm", {}), quantiles)

    def _fit_xgboost(
        self, X: np.ndarray, y: np.ndarray | None, params: dict, quantiles: list[float]
    ) -> None:
        import xgboost as xgb

        base_params = {
            "n_estimators": params.get("n_estimators", 500),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_weight": params.get("min_child_weight", 5),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "tree_method": params.get("tree_method", "hist"),
        }

        # Median model
        self.model_median = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles[1],
            **base_params,
        )
        self.model_median.fit(X, y)

        # Lower quantile
        self.model_lower = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles[0],
            **base_params,
        )
        self.model_lower.fit(X, y)

        # Upper quantile
        self.model_upper = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles[2],
            **base_params,
        )
        self.model_upper.fit(X, y)

    def _fit_lightgbm(
        self, X: np.ndarray, y: np.ndarray | None, params: dict, quantiles: list[float]
    ) -> None:
        import lightgbm as lgb

        base_params = {
            "n_estimators": params.get("n_estimators", 500),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_samples": params.get("min_child_samples", 20),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "verbose": -1,
        }

        self.model_median = lgb.LGBMRegressor(
            objective="quantile", alpha=quantiles[1], **base_params
        )
        self.model_median.fit(X, y)

        self.model_lower = lgb.LGBMRegressor(
            objective="quantile", alpha=quantiles[0], **base_params
        )
        self.model_lower.fit(X, y)

        self.model_upper = lgb.LGBMRegressor(
            objective="quantile", alpha=quantiles[2], **base_params
        )
        self.model_upper.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._prepare_xy(df, fit_encoders=False)
        try:
            return self.model_median.predict(X)
        except AttributeError:
            raise RuntimeError("Model has not been fitted. Call fit() first.") from None

    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        X, _ = self._prepare_xy(df, fit_encoders=False)
        try:
            lower = self.model_lower.predict(X)
            upper = self.model_upper.predict(X)
        except AttributeError:
            raise RuntimeError("Model has not been fitted. Call fit() first.") from None
        return lower, upper

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_median": self.model_median,
                    "model_lower": self.model_lower,
                    "model_upper": self.model_upper,
                    "label_encoders": self.label_encoders,
                    "feature_cols": self.feature_cols,
                    "categorical_cols": self.categorical_cols,
                    "config": self.config,
                    "backend": self.backend,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "GBMDegradationModel":
        instance = cls()
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        instance.model_median = data["model_median"]
        instance.model_lower = data["model_lower"]
        instance.model_upper = data["model_upper"]
        instance.label_encoders = data["label_encoders"]
        instance.feature_cols = data["feature_cols"]
        instance.categorical_cols = data["categorical_cols"]
        instance.config = data["config"]
        instance.backend = data["backend"]
        return instance

    def _prepare_xy(
        self, df: pd.DataFrame, fit_encoders: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        available_features = [c for c in self.feature_cols if c in df.columns]
        X_numeric = df[available_features].fillna(0).values.astype(float)

        available_cats = [c for c in self.categorical_cols if c in df.columns]
        cat_arrays = []
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
                # Handle unseen categories
                encoded = np.array([le.transform([v])[0] if v in le.classes_ else 0 for v in vals])
            cat_arrays.append(encoded.reshape(-1, 1))

        X = np.hstack([X_numeric, *cat_arrays]) if cat_arrays else X_numeric

        y = df["lap_time_seconds"].values if "lap_time_seconds" in df.columns else None
        return X, y
