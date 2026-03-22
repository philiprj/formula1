#!/usr/bin/env python3
"""Train the anomaly/retirement prediction model.

Uses the full dataset (laps_full.parquet) which includes outlier flags
and retirement labels as training targets.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1deg.config import load_config
from f1deg.eval.metrics import compute_classification_metrics
from f1deg.models.anomaly import AnomalyPredictionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def leave_one_race_out_cv(
    df: pd.DataFrame,
    config: dict,
    max_folds: int | None = None,
) -> dict:
    """Evaluate anomaly model with leave-one-race-out cross-validation."""
    race_ids = sorted(df["race_id"].unique())
    if max_folds and max_folds < len(race_ids):
        rng = np.random.default_rng(42)
        race_ids = rng.choice(race_ids, max_folds, replace=False).tolist()

    all_y_true = []
    all_y_proba = []

    for i, test_race in enumerate(race_ids):
        train = df[df["race_id"] != test_race]
        test = df[df["race_id"] == test_race]

        if test["is_anomalous_lap"].sum() == 0:
            # Skip races with no positive examples in test set
            continue

        model = AnomalyPredictionModel()
        model.fit(train, config)

        y_proba = model.predict_proba(test)
        y_true = test["is_anomalous_lap"].values.astype(int)

        all_y_true.append(y_true)
        all_y_proba.append(y_proba)

        if (i + 1) % 10 == 0:
            logger.info(f"  Completed fold {i + 1}/{len(race_ids)}")

    y_true = np.concatenate(all_y_true)
    y_proba = np.concatenate(all_y_proba)

    return compute_classification_metrics(y_true, y_proba)


def main():
    config = load_config()
    data_dir = Path(config["data_dir"])

    # Load model-specific config
    model_config_path = Path("conf/models/anomaly.yaml")
    if model_config_path.exists():
        import yaml

        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        config["model"] = model_config
    else:
        logger.warning("anomaly.yaml not found, using defaults")

    # Load full dataset
    full_path = data_dir / "processed" / "laps_full.parquet"
    if not full_path.exists():
        logger.error(
            f"{full_path} not found. Run `make features` first to generate "
            "the full dataset with outlier flags and retirement labels."
        )
        return

    logger.info(f"Loading full dataset from {full_path}...")
    df = pd.read_parquet(full_path)
    logger.info(f"Loaded {len(df)} laps")

    # Check for target column
    if "is_anomalous_lap" not in df.columns:
        logger.error(
            "is_anomalous_lap column not found. Ensure retirement labels "
            "have been added during feature building."
        )
        return

    anomalous_count = df["is_anomalous_lap"].sum()
    logger.info(
        f"Target distribution: {anomalous_count} anomalous "
        f"({anomalous_count / len(df) * 100:.1f}%), "
        f"{len(df) - anomalous_count} normal"
    )

    # Cross-validation
    logger.info("Running leave-one-race-out cross-validation...")
    cv_metrics = leave_one_race_out_cv(df, config, max_folds=20)
    logger.info("Cross-validation results:")
    for name, value in cv_metrics.items():
        logger.info(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    # Train final model on all data
    logger.info("Training final model on all data...")
    model = AnomalyPredictionModel()
    model.fit(df, config)

    # Feature importance
    importance = model.feature_importance()
    logger.info("Top 10 features by importance:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']}")

    # Save
    model_dir = data_dir / "models" / "anomaly"
    model.save(model_dir)
    logger.info(f"Saved anomaly model to {model_dir}")


if __name__ == "__main__":
    main()
