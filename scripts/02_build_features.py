#!/usr/bin/env python3
"""Build features from raw lap data."""

import logging
from pathlib import Path

from f1deg.config import load_config
from f1deg.data.features import build_features, load_raw_laps, save_features
from f1deg.data.filters import apply_filters
from f1deg.data.schemas import check_data_quality, validate_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    data_dir = Path(config["data_dir"])

    # Load raw data
    logger.info("Loading raw lap data...")
    raw_df = load_raw_laps(data_dir / "raw")
    logger.info(f"Loaded {len(raw_df)} raw laps")

    # Apply filters
    logger.info("Applying filters...")
    filter_config = config.get("features", {})
    filter_names = filter_config.get("filters")
    filtered_df = apply_filters(raw_df, filter_names=filter_names, config=config)

    # Build features
    logger.info("Building features...")
    features_df = build_features(filtered_df, config)

    # Validate
    logger.info("Validating processed data...")
    features_df = validate_processed(features_df)

    # Quality checks
    report = check_data_quality(features_df)
    logger.info(
        f"Data quality: {report['total_laps']} laps, {len(report.get('warnings', []))} warnings"
    )

    # Save
    output_path = save_features(features_df, data_dir / "processed")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
