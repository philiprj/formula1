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
        self.pi_scale_factor: float = 1.0  # Conformal calibration factor

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
        assert y is not None, "Training data must contain 'lap_time_seconds'"

        # Split into train + validation for early stopping & PI calibration.
        # Use race_id groups so entire races stay together (no within-race leak).
        val_frac = model_config.get("validation_fraction", 0.15)
        early_stopping = model_config.get("early_stopping_rounds", 50)

        if val_frac > 0 and "race_id" in train_df.columns:
            from sklearn.model_selection import GroupShuffleSplit

            groups = train_df["race_id"].values
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            early_stopping = None

        if self.backend == "xgboost":
            self._fit_xgboost(
                X_train,
                y_train,
                model_config.get("xgboost", {}),
                quantiles,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping,
            )
        else:
            self._fit_lightgbm(
                X_train,
                y_train,
                model_config.get("lightgbm", {}),
                quantiles,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping,
            )

        # Conformal PI calibration: scale intervals to achieve target coverage
        if X_val is not None and y_val is not None:
            self._calibrate_pi(X_val, y_val, target_coverage=0.95)

    def _fit_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        params: dict,
        quantiles: list[float],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int | None = 50,
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
        if early_stopping_rounds and X_val is not None:
            base_params["early_stopping_rounds"] = early_stopping_rounds

        fit_kwargs: dict = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        # Median model
        self.model_median = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles[1],
            **base_params,
        )
        self.model_median.fit(X, y, **fit_kwargs)

        # Lower quantile
        self.model_lower = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles[0],
            **base_params,
        )
        self.model_lower.fit(X, y, **fit_kwargs)

        # Upper quantile
        self.model_upper = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles[2],
            **base_params,
        )
        self.model_upper.fit(X, y, **fit_kwargs)

    def _fit_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        params: dict,
        quantiles: list[float],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> None:
        import lightgbm as lgb

        base_params = {
            "n_estimators": params.get("n_estimators", 500),
            "num_leaves": params.get("num_leaves", 63),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_samples": params.get("min_child_samples", 20),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "verbose": -1,
        }
        if "min_data_in_leaf" in params:
            base_params["min_data_in_leaf"] = params["min_data_in_leaf"]

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None and early_stopping_rounds:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        self.model_median = lgb.LGBMRegressor(
            objective="quantile", alpha=quantiles[1], **base_params
        )
        self.model_median.fit(X, y, **fit_kwargs)

        self.model_lower = lgb.LGBMRegressor(
            objective="quantile", alpha=quantiles[0], **base_params
        )
        self.model_lower.fit(X, y, **fit_kwargs)

        self.model_upper = lgb.LGBMRegressor(
            objective="quantile", alpha=quantiles[2], **base_params
        )
        self.model_upper.fit(X, y, **fit_kwargs)

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

        # Apply conformal calibration scaling if available
        if self.pi_scale_factor != 1.0:
            median = self.model_median.predict(X)
            lower = median - (median - lower) * self.pi_scale_factor
            upper = median + (upper - median) * self.pi_scale_factor

        return lower, upper

    def _calibrate_pi(
        self, X_val: np.ndarray, y_val: np.ndarray, target_coverage: float = 0.95
    ) -> None:
        """Post-hoc conformal calibration of prediction intervals.

        Scales the quantile-based intervals so that empirical coverage on a
        held-out set matches the target coverage.
        """
        pred_median = self.model_median.predict(X_val)
        pred_lower = self.model_lower.predict(X_val)
        pred_upper = self.model_upper.predict(X_val)

        # Current half-widths
        half_lower = pred_median - pred_lower
        half_upper = pred_upper - pred_median

        # For each sample, find the scale factor needed to cover it
        residuals = y_val - pred_median
        # How much we'd need to scale the interval to cover each point
        with np.errstate(divide="ignore", invalid="ignore"):
            scale_needed = np.where(
                residuals < 0,
                np.where(half_lower > 1e-6, np.abs(residuals) / half_lower, np.inf),
                np.where(half_upper > 1e-6, np.abs(residuals) / half_upper, np.inf),
            )
        # The scale factor at the target quantile gives us the desired coverage
        # Cap individual scales to avoid extreme outliers inflating the factor
        finite_scales = scale_needed[np.isfinite(scale_needed) & (scale_needed < 20.0)]
        if len(finite_scales) > 10:
            self.pi_scale_factor = float(np.quantile(finite_scales, target_coverage))
        else:
            self.pi_scale_factor = 1.0

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
                    "pi_scale_factor": self.pi_scale_factor,
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
        instance.pi_scale_factor = data.get("pi_scale_factor", 1.0)
        return instance

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Compute rolling/lag features within each stint for degradation modeling.

        These are computed at prepare time (not in the feature pipeline) to avoid
        CV leakage — each fold gets rolling stats from its own data only.
        """
        result = df.copy()

        group_cols = []
        if "race_id" in result.columns:
            group_cols.append("race_id")
        if "driver_id" in result.columns:
            group_cols.append("driver_id")
        if "stint_number" in result.columns:
            group_cols.append("stint_number")

        if not group_cols or "lap_time_seconds" not in result.columns:
            return result

        time_col = "lap_time_seconds"
        grouped = result.groupby(group_cols)[time_col]

        # lap_time_delta: difference from rolling median
        rolling_median = grouped.transform(lambda x: x.rolling(window, min_periods=2).median())
        result["lap_time_delta"] = result[time_col] - rolling_median

        # lap_time_rolling_std: rolling standard deviation
        result["lap_time_rolling_std"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )

        # deg_rate_estimate: lap-over-lap time increase within stint
        result["deg_rate_estimate"] = grouped.diff()

        # tyre_life_cubed: cubic term for super-quadratic degradation
        if "tyre_life" in result.columns:
            result["tyre_life_cubed"] = result["tyre_life"] ** 3

        # Fill NaN from cold-start (first laps in stint)
        for col in ["lap_time_delta", "lap_time_rolling_std", "deg_rate_estimate"]:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)

        return result

    def _prepare_xy(
        self, df: pd.DataFrame, fit_encoders: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # Build numeric features — use exactly self.feature_cols in order,
        # filling any missing columns with 0 so the feature count always
        # matches the trained model.
        df_prep = self._add_rolling_features(df)
        for col in self.feature_cols:
            if col not in df_prep.columns:
                df_prep[col] = 0.0
        X_numeric = df_prep[self.feature_cols].fillna(0).values.astype(float)

        # Encode categoricals — use exactly self.categorical_cols in order
        cat_arrays = []
        for col in self.categorical_cols:
            vals = (
                df_prep[col].fillna("UNKNOWN").astype(str)
                if col in df_prep.columns
                else pd.Series(["UNKNOWN"] * len(df_prep))
            )
            if fit_encoders:
                le = LabelEncoder()
                encoded = le.fit_transform(vals)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    encoded = np.zeros(len(vals), dtype=int)
                else:
                    encoded = np.array(
                        [le.transform([v])[0] if v in le.classes_ else 0 for v in vals]
                    )
            cat_arrays.append(encoded.reshape(-1, 1))

        X = np.hstack([X_numeric, *cat_arrays]) if cat_arrays else X_numeric

        y = df["lap_time_seconds"].values if "lap_time_seconds" in df.columns else None
        return X, y
