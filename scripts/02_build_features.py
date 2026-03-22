#!/usr/bin/env python3
"""Build features from raw lap data.

Produces two outputs:
  - laps_clean.parquet: Clean laps only (outliers removed), for degradation models.
  - laps_full.parquet: All laps with outlier/retirement labels, for anomaly model.
"""

import logging
from pathlib import Path

from f1deg.config import load_config
from f1deg.data.features import build_features, load_raw_laps, save_features
from f1deg.data.filters import (
    apply_filters,
    flag_outliers_compound_aware,
    flag_yellow_adjacent,
)
from f1deg.data.labels import add_retirement_labels
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

    # Apply basic filters (accurate, track_status, pit_laps, first_lap)
    # Note: we do NOT apply the old "outliers" filter here — we flag instead
    logger.info("Applying basic filters...")
    basic_filters = ["accurate", "track_status", "pit_laps", "first_lap"]
    filtered_df = apply_filters(raw_df, filter_names=basic_filters, config=config)

    # Build features
    logger.info("Building features...")
    features_df = build_features(filtered_df, config)

    # Flag outliers (compound-aware z-score + yellow adjacency)
    logger.info("Flagging outliers...")
    features_df = flag_outliers_compound_aware(features_df)
    features_df = flag_yellow_adjacent(features_df)
    outlier_count = features_df["is_outlier"].sum()
    logger.info(f"Flagged {outlier_count}/{len(features_df)} laps as outliers")

    # Add retirement labels from Jolpica results
    results_dir = data_dir / "results"
    if results_dir.exists():
        logger.info("Adding retirement labels...")
        features_df = add_retirement_labels(features_df, results_dir)
    else:
        logger.warning(
            f"Results directory {results_dir} not found. "
            "Run results ingestion first for retirement labels."
        )

    # Validate
    logger.info("Validating processed data...")
    features_df = validate_processed(features_df)

    # Quality checks
    report = check_data_quality(features_df)
    logger.info(
        f"Data quality: {report['total_laps']} laps, {len(report.get('warnings', []))} warnings"
    )

    # Save full dataset (all laps with outlier flags)
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "laps_full.parquet"
    features_df.to_parquet(full_path, index=False)
    logger.info(f"Saved full dataset ({len(features_df)} laps) to {full_path}")

    # Save clean dataset (outliers removed, backward compatible)
    clean_df = features_df[~features_df.get("is_outlier", False)].copy()
    # Drop outlier/label columns from clean dataset
    drop_cols = [
        c
        for c in [
            "is_outlier",
            "outlier_reason",
            "is_anomalous_lap",
            "did_retire",
            "laps_until_retirement",
            "retirement_lap",
        ]
        if c in clean_df.columns
    ]
    clean_df = clean_df.drop(columns=drop_cols)
    output_path = save_features(clean_df, output_dir)
    logger.info(f"Saved clean dataset ({len(clean_df)} laps) to {output_path}")


if __name__ == "__main__":
    main()
