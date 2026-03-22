#!/usr/bin/env python3
"""Train a tire degradation model."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from f1deg.config import load_config
from f1deg.models import MODEL_REGISTRY, get_model_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a tire degradation model")
    parser.add_argument(
        "model",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to train",
    )
    parser.add_argument(
        "--svi", action="store_true", help="Use SVI instead of MCMC (Bayesian only)"
    )
    args = parser.parse_args()

    config = load_config(model_name=args.model)
    data_dir = Path(config["data_dir"])

    # Load processed data
    data_path = data_dir / "processed" / "laps_clean.parquet"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} laps")

    # Train model
    model_cls = get_model_class(args.model)
    model = model_cls()

    if args.svi and hasattr(model, "fit_svi"):
        logger.info(f"Training {args.model} model with SVI...")
        model.fit_svi(df, config)
    else:
        logger.info(f"Training {args.model} model...")
        model.fit(df, config)

    # Save
    model_dir = data_dir / "models" / args.model
    model.save(model_dir)
    logger.info(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
