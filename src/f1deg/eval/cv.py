"""Leave-one-race-out cross-validation for tire degradation models."""

import logging

import numpy as np
import pandas as pd

from f1deg.eval.metrics import compute_all_metrics
from f1deg.models.base import DegradationModel

logger = logging.getLogger(__name__)


def leave_one_race_out_cv(
    df: pd.DataFrame,
    model_cls: type[DegradationModel],
    config: dict,
    max_folds: int | None = None,
    use_svi: bool = False,
) -> dict:
    """Run leave-one-race-out cross-validation.

    Args:
        df: Full processed dataset.
        model_cls: Model class to instantiate for each fold.
        config: Model configuration.
        max_folds: If set, randomly sample this many folds (for speed).
        use_svi: If True and model has fit_svi, use variational inference.

    Returns:
        Dict with per-fold and aggregate metrics.
    """
    if "race_id" not in df.columns:
        raise ValueError("DataFrame must contain 'race_id' column for LORO CV")

    race_ids = sorted(df["race_id"].unique())

    if max_folds and max_folds < len(race_ids):
        rng = np.random.default_rng(42)
        race_ids = rng.choice(race_ids, size=max_folds, replace=False).tolist()

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_lower = []
    all_upper = []

    for i, test_race in enumerate(race_ids):
        logger.info(f"Fold {i + 1}/{len(race_ids)}: testing on {test_race}")

        train_df = df[df["race_id"] != test_race].copy()
        test_df = df[df["race_id"] == test_race].copy()

        if len(test_df) < 5:
            logger.warning(f"Skipping {test_race}: only {len(test_df)} test laps")
            continue

        model = model_cls()

        # Use SVI for Bayesian models during CV (much faster)
        if use_svi and hasattr(model, "fit_svi"):
            model.fit_svi(train_df, config)
        else:
            model.fit(train_df, config)

        y_true = test_df["lap_time_seconds"].values
        y_pred = model.predict(test_df)
        lower, upper = model.predict_interval(test_df)

        fold_metrics = compute_all_metrics(y_true, y_pred, lower, upper, test_df)
        fold_metrics["race_id"] = test_race
        fold_metrics["n_test_laps"] = len(test_df)
        fold_results.append(fold_metrics)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_lower.append(lower)
        all_upper.append(upper)

        logger.info(
            f"  MAE: {fold_metrics['mae']:.3f}s, PI coverage: {fold_metrics.get('pi_coverage_95', 'N/A')}"
        )

    # Aggregate metrics
    if not fold_results:
        return {"fold_results": [], "aggregate": {}}

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    lower_all = np.concatenate(all_lower)
    upper_all = np.concatenate(all_upper)

    aggregate = compute_all_metrics(y_true_all, y_pred_all, lower_all, upper_all)
    aggregate["n_folds"] = len(fold_results)
    aggregate["n_total_laps"] = len(y_true_all)

    # Per-fold summary stats
    fold_maes = [f["mae"] for f in fold_results]
    aggregate["mae_std"] = float(np.std(fold_maes))
    aggregate["mae_min"] = float(np.min(fold_maes))
    aggregate["mae_max"] = float(np.max(fold_maes))

    return {
        "fold_results": fold_results,
        "aggregate": aggregate,
    }
