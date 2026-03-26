#!/usr/bin/env python3
"""Evaluate a trained model on holdout races (no retraining).

Loads the already-trained model from disk and scores it against the most
recent races as a holdout set.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1deg.config import load_config
from f1deg.eval.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained tire degradation model")
    parser.add_argument("model", help="Model type (linear, bayesian, gbm, sequence)")
    parser.add_argument(
        "--holdout-races",
        type=int,
        default=5,
        help="Number of most recent races to hold out (default: 5)",
    )
    args = parser.parse_args()

    config = load_config(model_name=args.model)
    data_dir = Path(config["data_dir"])

    # Check trained model exists
    model_dir = data_dir / "models" / args.model
    if not model_dir.exists():
        logger.error(f"No trained model at {model_dir}. Run 'make train MODEL={args.model}' first.")
        raise SystemExit(1)

    # Load trained model
    from f1deg.models import get_model_class

    model_cls = get_model_class(args.model)
    logger.info(f"Loading trained {args.model} model from {model_dir}")
    model = model_cls.load(model_dir)

    # Load data
    df = pd.read_parquet(data_dir / "processed" / "laps_clean.parquet")
    race_ids = sorted(df["race_id"].unique())
    logger.info(f"Loaded {len(df)} laps, {len(race_ids)} races")

    # Hold out most recent races
    holdout_races = min(args.holdout_races, max(1, len(race_ids) // 5))
    test_races = race_ids[-holdout_races:]
    test_df = df[df["race_id"].isin(test_races)].copy()
    logger.info(
        f"Holdout: {len(test_races)} races ({test_races[0]} .. {test_races[-1]}), "
        f"{len(test_df)} laps"
    )

    # Predict
    y_true = test_df["lap_time_seconds"].values
    y_pred = model.predict(test_df)
    lower, upper = model.predict_interval(test_df)

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, lower, upper, test_df)
    metrics["n_holdout_races"] = len(test_races)
    metrics["n_holdout_laps"] = len(test_df)

    # Print summary
    print(f"\n=== Eval: {args.model} ({len(test_races)} holdout races) ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            if np.isnan(value):
                print(f"  {key}: N/A")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
