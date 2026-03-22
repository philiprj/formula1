"""Retirement and anomaly labeling for lap data.

Adds per-lap labels for use by the anomaly prediction model:
  - did_retire: whether the driver retired from this race
  - retirement_lap: last lap completed before retirement
  - laps_until_retirement: countdown (null for finishers)
  - is_anomalous_lap: outlier OR within N laps of retirement
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of laps before retirement to label as anomalous
RETIREMENT_PROXIMITY_LAPS = 3


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all results Parquet files from the results directory."""
    files = sorted(results_dir.glob("results_*.parquet"))
    if not files:
        logger.warning(f"No results files found in {results_dir}")
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} results from {len(files)} files")
    return combined


def add_retirement_labels(
    df: pd.DataFrame,
    results_dir: Path,
    proximity_laps: int = RETIREMENT_PROXIMITY_LAPS,
) -> pd.DataFrame:
    """Add retirement/anomaly labels to a lap DataFrame.

    Args:
        df: Lap DataFrame with race_id, driver_id, lap_number columns.
        results_dir: Directory containing results_YYYY.parquet files.
        proximity_laps: Number of laps before retirement to flag as anomalous.

    Returns:
        DataFrame with added label columns.
    """
    results = load_results(results_dir)
    if results.empty:
        logger.warning("No results data available, skipping retirement labels")
        df["did_retire"] = False
        df["retirement_lap"] = np.nan
        df["laps_until_retirement"] = np.nan
        df["is_anomalous_lap"] = df.get("is_outlier", False)
        return df

    # Build race_id in results to match laps
    if "year" in results.columns and "round" in results.columns:
        results["race_id"] = (
            results["year"].astype(str) + "_" + results["round"].astype(str).str.zfill(2)
        )

    # Create a lookup: (race_id, driver_id) -> {did_retire, laps_completed}
    retirement_lookup = {}
    for _, row in results.iterrows():
        key = (row.get("race_id", ""), row.get("driver_id", ""))
        retirement_lookup[key] = {
            "did_retire": bool(row.get("did_retire", False)),
            "laps_completed": int(row.get("laps_completed", 0)),
        }

    result = df.copy()
    result["did_retire"] = False
    result["retirement_lap"] = np.nan
    result["laps_until_retirement"] = np.nan

    for idx, row in result.iterrows():
        key = (row.get("race_id", ""), row.get("driver_id", ""))
        info = retirement_lookup.get(key)
        if info and info["did_retire"]:
            result.loc[idx, "did_retire"] = True
            result.loc[idx, "retirement_lap"] = info["laps_completed"]
            laps_until = info["laps_completed"] - row.get("lap_number", 0)
            if laps_until >= 0:
                result.loc[idx, "laps_until_retirement"] = laps_until

    # is_anomalous_lap: outlier OR close to retirement
    is_outlier = result.get("is_outlier", pd.Series(False, index=result.index))
    near_retirement = (
        result["laps_until_retirement"].notna()
        & (result["laps_until_retirement"] <= proximity_laps)
        & (result["laps_until_retirement"] >= 0)
    )
    result["is_anomalous_lap"] = is_outlier | near_retirement

    retire_count = result.groupby(["race_id", "driver_id"])["did_retire"].first().sum()
    anomalous_count = result["is_anomalous_lap"].sum()
    logger.info(
        f"Retirement labels: {retire_count} driver-retirements, "
        f"{anomalous_count} anomalous laps flagged"
    )

    return result
