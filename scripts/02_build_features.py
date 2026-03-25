#!/usr/bin/env python3
"""Build features from raw lap data.

Produces two outputs:
  - laps_clean.parquet: Clean laps only (outliers removed), for degradation models.
  - laps_full.parquet: All laps with outlier/retirement labels, for anomaly model.
"""

import logging
from pathlib import Path

from f1deg.config import load_config
from f1deg.data.features import (
    build_features,
    compute_sc_rain_features,
    load_raw_laps,
    save_features,
)
from f1deg.data.filters import (
    apply_filters,
    flag_outliers_compound_aware,
    flag_yellow_adjacent,
)
from f1deg.data.labels import add_retirement_labels
from f1deg.data.schemas import check_data_quality, validate_processed
from f1deg.data.weekend import build_weekend_calibration, fill_weekend_features

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

    # Compute SC/rain proximity features from FULL raw data (before filtering
    # removes SC/VSC/Red flag laps — we need the complete TrackStatus sequence)
    logger.info("Computing SC/rain proximity features from raw data...")
    sc_rain_df = compute_sc_rain_features(raw_df)
    if not sc_rain_df.empty:
        logger.info(f"SC/rain features: {len(sc_rain_df)} laps computed")

    # Apply basic filters (accurate, track_status, pit_laps, first_lap)
    # Note: we do NOT apply the old "outliers" filter here — we flag instead
    logger.info("Applying basic filters...")
    basic_filters = ["accurate", "track_status", "pit_laps", "first_lap"]
    filtered_df = apply_filters(raw_df, filter_names=basic_filters, config=config)

    # Build features
    logger.info("Building features...")
    features_df = build_features(filtered_df, config)

    # Merge SC/rain features onto the filtered dataset
    if not sc_rain_df.empty and "race_id" in features_df.columns:
        merge_cols = ["race_id", "driver_id", "lap_number"]
        sc_feature_cols = [c for c in sc_rain_df.columns if c not in merge_cols]
        features_df = features_df.merge(
            sc_rain_df[merge_cols + sc_feature_cols],
            on=merge_cols,
            how="left",
        )
        # Fill any unmatched laps with safe defaults
        features_df["laps_since_sc_end"] = features_df["laps_since_sc_end"].fillna(5)
        features_df["laps_since_red_flag"] = features_df["laps_since_red_flag"].fillna(5)
        features_df["had_sc_this_stint"] = features_df["had_sc_this_stint"].fillna(False)
        features_df["compound_class"] = features_df["compound_class"].fillna("dry")
        features_df["is_wet_running"] = features_df["is_wet_running"].fillna(0.0)
        features_df["compound_class_changed_this_stint"] = features_df[
            "compound_class_changed_this_stint"
        ].fillna(0.0)
        features_df["laps_since_compound_class_change"] = features_df[
            "laps_since_compound_class_change"
        ].fillna(10)
        features_df["sub_race_id"] = features_df["sub_race_id"].fillna(1)
        logger.info(f"Merged SC/rain features onto {len(features_df)} laps")

    # Merge weekend calibration features from practice/qualifying sessions
    raw_dir = data_dir / "raw"
    if "race_id" in features_df.columns:
        race_ids = sorted(features_df["race_id"].unique())
        logger.info(f"Building weekend calibration features for {len(race_ids)} races...")
        weekend_df = build_weekend_calibration(raw_dir, race_ids)
        if not weekend_df.empty:
            features_df = features_df.merge(weekend_df, on=["race_id", "driver_id"], how="left")
            logger.info(f"Merged weekend features: {len(weekend_df)} driver-race entries")
        # Fill missing values with hierarchical fallbacks
        features_df = fill_weekend_features(features_df)

    # Flag outliers (compound-aware z-score + yellow adjacency)
    logger.info("Flagging outliers...")
    features_df = flag_outliers_compound_aware(features_df)
    # Post-SC laps are now handled by laps_since_sc_end feature,
    # so we only flag pre-SC adjacent laps (not post-SC).
    features_df = flag_yellow_adjacent(features_df, adjacent_laps=1, sc_adjacent_laps=2)
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

    # Split into dry and wet datasets
    dry_compounds = {"SOFT", "MEDIUM", "HARD"}
    dry_df = clean_df[clean_df["compound"].isin(dry_compounds)].copy()
    wet_df = clean_df[~clean_df["compound"].isin(dry_compounds)].copy()

    # Save dry-only as laps_clean.parquet (main training data)
    output_path = save_features(dry_df, output_dir)
    logger.info(f"Saved dry-only clean dataset ({len(dry_df)} laps) to {output_path}")

    # Save wet laps separately for future wet-specific modeling
    if len(wet_df) > 0:
        wet_path = output_dir / "laps_wet.parquet"
        wet_df.to_parquet(wet_path, index=False)
        logger.info(f"Saved wet dataset ({len(wet_df)} laps) to {wet_path}")


if __name__ == "__main__":
    main()
