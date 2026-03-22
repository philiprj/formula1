#!/usr/bin/env python3
"""Evaluate a trained model with leave-one-race-out CV."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from f1deg.config import load_config
from f1deg.eval.cv import leave_one_race_out_cv
from f1deg.eval.report import generate_cv_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a tire degradation model")
    parser.add_argument("model", help="Model type (linear, bayesian, gbm, sequence)")
    parser.add_argument("--max-folds", type=int, default=None, help="Limit number of CV folds")
    parser.add_argument("--svi", action="store_true", help="Use SVI for Bayesian CV")
    args = parser.parse_args()

    config = load_config(model_name=args.model)
    data_dir = Path(config["data_dir"])

    # Load data
    df = pd.read_parquet(data_dir / "processed" / "laps_clean.parquet")
    logger.info(f"Loaded {len(df)} laps, {df['race_id'].nunique()} races")

    # Import model class
    from f1deg.models import get_model_class

    model_cls = get_model_class(args.model)

    # Run CV
    cv_results = leave_one_race_out_cv(
        df=df,
        model_cls=model_cls,
        config=config,
        max_folds=args.max_folds,
        use_svi=args.svi,
    )

    # Generate report
    report_dir = data_dir / "models" / args.model
    report_path = generate_cv_report(cv_results, args.model, report_dir)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    agg = cv_results.get("aggregate", {})
    print("\n=== CV Results ===")
    for key, value in sorted(agg.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
