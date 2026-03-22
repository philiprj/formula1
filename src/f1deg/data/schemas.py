"""DataFrame schema validation for raw and processed data."""

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

logger = logging.getLogger(__name__)

# Schema for processed/feature-engineered data
PROCESSED_SCHEMA = DataFrameSchema(
    columns={
        "race_id": Column(str, nullable=False),
        "circuit_id": Column(str, nullable=False),
        "driver_id": Column(str, nullable=False),
        "constructor_id": Column(str, nullable=False),
        "lap_number": Column(int, pa.Check.gt(0)),
        "lap_time_seconds": Column(float, pa.Check.in_range(60.0, 200.0)),
        "tyre_life": Column(float, pa.Check.ge(0)),
        "tyre_life_sq": Column(float, pa.Check.ge(0)),
        "compound": Column(
            str,
            pa.Check.isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
        ),
        "fuel_mass_kg": Column(float, pa.Check.in_range(0.0, 115.0)),
        "stint_number": Column(int, pa.Check.ge(0)),
        "stint_lap": Column(int, pa.Check.ge(0)),
        # Traffic/position features (optional — may be absent in older data)
        "position": Column(float, pa.Check.in_range(1, 22), nullable=True, required=False),
        "position_change": Column(float, nullable=True, required=False),
        "gap_ahead_seconds": Column(float, pa.Check.ge(0), nullable=True, required=False),
        "gap_behind_seconds": Column(float, pa.Check.ge(0), nullable=True, required=False),
        "traffic_density": Column(int, pa.Check.ge(0), nullable=True, required=False),
        # Stint context features
        "race_progress": Column(float, pa.Check.in_range(0.0, 1.05), nullable=True, required=False),
        "stint_fraction": Column(
            float, pa.Check.in_range(0.0, 1.05), nullable=True, required=False
        ),
        "is_final_stint": Column(bool, nullable=True, required=False),
        # Interaction features
        "compound_x_track_temp": Column(float, nullable=True, required=False),
        "tyre_life_x_track_temp": Column(float, nullable=True, required=False),
        # Outlier / anomaly labels (present in full dataset)
        "is_outlier": Column(bool, nullable=True, required=False),
        "outlier_reason": Column(str, nullable=True, required=False),
        "is_anomalous_lap": Column(bool, nullable=True, required=False),
        "did_retire": Column(bool, nullable=True, required=False),
        "retirement_lap": Column(float, nullable=True, required=False),
        "laps_until_retirement": Column(float, nullable=True, required=False),
    },
    # Weather columns and other optional columns
    strict=False,
    coerce=True,
)


def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Validate processed DataFrame against schema.

    Returns validated DataFrame (with coerced types).
    Raises pandera.errors.SchemaError on validation failure.
    """
    logger.info(f"Validating processed DataFrame: {len(df)} rows, {len(df.columns)} columns")
    validated = PROCESSED_SCHEMA.validate(df)
    logger.info("Validation passed")
    return validated


def check_data_quality(df: pd.DataFrame) -> dict:
    """Run data quality checks and return a summary report.

    Non-blocking — logs warnings but doesn't raise.
    """
    report = {
        "total_laps": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "warnings": [],
    }

    # Check for reasonable lap time distribution
    if "lap_time_seconds" in df.columns:
        mean_time = df["lap_time_seconds"].mean()
        if mean_time < 70 or mean_time > 120:
            report["warnings"].append(f"Unusual mean lap time: {mean_time:.1f}s (expected 70-120s)")

    # Check compound distribution
    if "compound" in df.columns:
        compound_counts = df["compound"].value_counts()
        report["compound_distribution"] = compound_counts.to_dict()
        if len(compound_counts) < 2:
            report["warnings"].append("Only one compound type found")

    # Check for duplicates
    dup_cols = ["race_id", "driver_id", "lap_number"]
    if all(c in df.columns for c in dup_cols):
        n_dups = df.duplicated(subset=dup_cols).sum()
        if n_dups > 0:
            report["warnings"].append(f"{n_dups} duplicate (race, driver, lap) entries found")
        report["duplicates"] = int(n_dups)

    for warning in report["warnings"]:
        logger.warning(warning)

    return report
