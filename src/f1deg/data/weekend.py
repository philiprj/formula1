"""Weekend calibration features from practice and qualifying sessions.

Aggregates FP1/FP2/FP3/Q laps into per-(race_id, driver_id) summary features
that calibrate the race model to this specific weekend's conditions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1deg.data.features import compute_lap_time_seconds, compute_stint_info

logger = logging.getLogger(__name__)

# Weekend features that will be added to the race dataset
WEEKEND_FEATURE_COLS = [
    "circuit_baseline_pace",
    "driver_pace_vs_field",
    "fp_deg_rate_soft",
    "fp_deg_rate_medium",
    "fp_deg_rate_hard",
    "weekend_track_temp",
    "quali_position",
    "expected_fp3_race_delta",
]

# Typical compound-to-compound pace offset (seconds per lap, approximate)
# Used as fallback when a compound's deg rate is missing
_COMPOUND_DEG_RATIO = {"SOFT": 1.3, "MEDIUM": 1.0, "HARD": 0.75}


def load_practice_laps(raw_dir: Path, race_id: str) -> pd.DataFrame:
    """Load FP1/FP2/FP3 parquet files for a single race weekend.

    Args:
        raw_dir: Directory containing raw parquet files.
        race_id: Race identifier in "{year}_{round:02d}" format.

    Returns:
        DataFrame of practice laps, or empty DataFrame if none found.
    """
    year, round_str = race_id.split("_")
    prefix = f"{year}_{round_str}_"

    practice_dfs = []
    for suffix in ["_fp1.parquet", "_fp2.parquet", "_fp3.parquet"]:
        matches = list(raw_dir.glob(f"{prefix}*{suffix}"))
        for path in matches:
            df = pd.read_parquet(path)
            practice_dfs.append(df)

    if not practice_dfs:
        return pd.DataFrame()

    return pd.concat(practice_dfs, ignore_index=True)


def load_qualifying_laps(raw_dir: Path, race_id: str) -> pd.DataFrame:
    """Load qualifying parquet file for a single race weekend."""
    year, round_str = race_id.split("_")
    prefix = f"{year}_{round_str}_"

    matches = list(raw_dir.glob(f"{prefix}*_q.parquet"))
    if not matches:
        return pd.DataFrame()

    return pd.read_parquet(matches[0])


def _filter_clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Filter practice/qualifying laps to keep only clean, representative laps."""
    result = df.copy()

    # Deduplicate: FastF1 practice sessions often contain duplicated rows
    dedup_cols = ["Driver", "LapNumber"]
    if all(c in result.columns for c in dedup_cols):
        result = result.drop_duplicates(subset=dedup_cols, keep="first")

    # Keep only accurate laps
    if "IsAccurate" in result.columns:
        result = result[result["IsAccurate"] == True]  # noqa: E712

    # Exclude safety car / red flag laps
    if "TrackStatus" in result.columns:
        clean_statuses = {"1", "2", 1, 2}  # 1=green, 2=yellow (local)
        result = result[result["TrackStatus"].isin(clean_statuses)]

    # Exclude pit in/out laps
    for col in ["PitInTime", "PitOutTime"]:
        if col in result.columns:
            result = result[result[col].isna()]

    # Remove slow laps: keep only laps within 107% of session best
    # This filters installation laps, cool-down laps, and traffic-affected laps
    time_col = None
    if "LapTime_seconds" in result.columns:
        time_col = "LapTime_seconds"
    elif "LapTime" in result.columns and pd.api.types.is_timedelta64_dtype(result["LapTime"]):
        result["_lap_time_s"] = result["LapTime"].dt.total_seconds()
        time_col = "_lap_time_s"

    if time_col and not result.empty:
        valid = result[result[time_col].notna()]
        if not valid.empty:
            session_best = valid[time_col].min()
            threshold = session_best * 1.07
            result = result[result[time_col] <= threshold]

    if "_lap_time_s" in result.columns:
        result = result.drop(columns=["_lap_time_s"])

    return result


def compute_deg_rate(
    stints_df: pd.DataFrame, compound: str, min_stint_laps: int = 5
) -> float | None:
    """Estimate degradation rate (s/lap) from long-run practice stints.

    Finds stints of the given compound with at least min_stint_laps,
    computes OLS slope of lap_time vs tyre_life for each, and returns
    the median slope.

    Returns None if no qualifying stints found.
    """
    if stints_df.empty:
        return None

    compound_df = stints_df[stints_df["compound"] == compound].copy()
    if compound_df.empty:
        return None

    # Group by driver + stint, keep only long runs
    group_cols = []
    if "driver_id" in compound_df.columns:
        group_cols.append("driver_id")
    if "stint_number" in compound_df.columns:
        group_cols.append("stint_number")

    if not group_cols:
        return None

    slopes = []
    for _, stint in compound_df.groupby(group_cols):
        if len(stint) < min_stint_laps:
            continue

        x = stint["tyre_life"].values
        y = stint["lap_time_seconds"].values

        if np.std(x) == 0:
            continue

        # OLS slope: cov(x,y) / var(x)
        slope = np.cov(x, y)[0, 1] / np.var(x)
        # Degradation must be positive and within realistic range
        # Typical F1 deg rates: 0.01-0.25 s/lap
        if np.isfinite(slope) and 0.001 < slope < 0.5:
            slopes.append(slope)

    return float(np.median(slopes)) if slopes else None


def compute_weekend_features(
    practice_df: pd.DataFrame,
    qualifying_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate practice/qualifying laps into per-(race_id, driver_id) features.

    Args:
        practice_df: Concatenated FP1/FP2/FP3 laps (raw format).
        qualifying_df: Optional qualifying laps (raw format).

    Returns:
        DataFrame with columns: race_id, driver_id, + weekend calibration features.
    """
    if practice_df.empty:
        return pd.DataFrame(columns=["race_id", "driver_id", *WEEKEND_FEATURE_COLS])

    # Prepare practice data
    fp = _filter_clean_laps(practice_df)
    if fp.empty:
        return pd.DataFrame(columns=["race_id", "driver_id", *WEEKEND_FEATURE_COLS])

    # Compute basic columns needed
    fp["lap_time_seconds"] = compute_lap_time_seconds(fp)
    fp["driver_id"] = fp["Driver"] if "Driver" in fp.columns else "UNKNOWN"

    if "Year" in fp.columns and "RoundNumber" in fp.columns:
        fp["race_id"] = fp["Year"].astype(str) + "_" + fp["RoundNumber"].astype(str).str.zfill(2)
    else:
        return pd.DataFrame(columns=["race_id", "driver_id", *WEEKEND_FEATURE_COLS])

    if "TyreLife" in fp.columns:
        fp["tyre_life"] = fp["TyreLife"].astype(float)
    if "Compound" in fp.columns:
        fp["compound"] = fp["Compound"].str.upper()

    # Identify FP3 laps (best for baseline pace — closest to race conditions)
    session_col = "SessionType" if "SessionType" in fp.columns else None
    is_fp3 = (
        fp[session_col].str.contains("3", na=False)
        if session_col
        else pd.Series(True, index=fp.index)
    )
    fp3 = fp[is_fp3]
    if fp3.empty:
        fp3 = fp  # Fall back to all practice data

    # Compute stint info for degradation rate estimation
    if all(c in fp.columns for c in ["TyreLife", "driver_id"]):
        fp_stint = compute_stint_info(
            fp.rename(columns={"driver_id": "Driver"}) if "Driver" not in fp.columns else fp
        )
        if "stint_number" not in fp.columns:
            fp["stint_number"] = fp_stint["stint_number"]

    results = []
    for race_id, race_fp in fp.groupby("race_id"):
        race_fp3 = fp3[fp3["race_id"] == race_id] if not fp3.empty else race_fp

        # Circuit baseline pace: median clean lap across all drivers in FP3
        circuit_baseline = race_fp3["lap_time_seconds"].median()

        for driver_id, driver_fp in race_fp.groupby("driver_id"):
            driver_fp3 = race_fp3[race_fp3["driver_id"] == driver_id]

            # Driver's FP3 pace
            driver_pace = (
                driver_fp3["lap_time_seconds"].median()
                if not driver_fp3.empty
                else driver_fp["lap_time_seconds"].median()
            )

            # Degradation rates from long runs (primarily FP2)
            deg_soft = compute_deg_rate(driver_fp, "SOFT")
            deg_medium = compute_deg_rate(driver_fp, "MEDIUM")
            deg_hard = compute_deg_rate(driver_fp, "HARD")

            # Weekend track temperature
            track_temp = race_fp["TrackTemp"].mean() if "TrackTemp" in race_fp.columns else None

            row = {
                "race_id": race_id,
                "driver_id": driver_id,
                "circuit_baseline_pace": circuit_baseline,
                "driver_pace_vs_field": driver_pace - circuit_baseline
                if circuit_baseline and driver_pace
                else None,
                "fp_deg_rate_soft": deg_soft,
                "fp_deg_rate_medium": deg_medium,
                "fp_deg_rate_hard": deg_hard,
                "weekend_track_temp": track_temp,
            }
            results.append(row)

    result_df = pd.DataFrame(results)

    # Add qualifying position if available
    if qualifying_df is not None and not qualifying_df.empty:
        qual = _compute_quali_position(qualifying_df)
        if not qual.empty:
            result_df = result_df.merge(qual, on=["race_id", "driver_id"], how="left")

    if "quali_position" not in result_df.columns:
        result_df["quali_position"] = np.nan

    return result_df


def _compute_quali_position(qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """Extract qualifying position per driver from qualifying laps."""
    df = qualifying_df.copy()
    df["lap_time_seconds"] = compute_lap_time_seconds(df)
    df["driver_id"] = df["Driver"] if "Driver" in df.columns else "UNKNOWN"

    if "Year" in df.columns and "RoundNumber" in df.columns:
        df["race_id"] = df["Year"].astype(str) + "_" + df["RoundNumber"].astype(str).str.zfill(2)
    else:
        return pd.DataFrame(columns=["race_id", "driver_id", "quali_position"])

    # Best lap time per driver
    best = df.groupby(["race_id", "driver_id"])["lap_time_seconds"].min().reset_index()
    best = best.sort_values(["race_id", "lap_time_seconds"])
    best["quali_position"] = best.groupby("race_id").cumcount() + 1

    return best[["race_id", "driver_id", "quali_position"]]


def fill_weekend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weekend calibration features with hierarchical fallbacks.

    Fallback levels:
        1. This weekend's practice data (already merged, may be NaN)
        2. Historical average for this circuit across prior seasons
        3. Global mean across all races
    """
    result = df.copy()

    for col in WEEKEND_FEATURE_COLS:
        if col not in result.columns:
            result[col] = np.nan
            continue

        # Level 2: Fill from circuit historical average
        if "circuit_id" in result.columns:
            circuit_means = result.groupby("circuit_id")[col].transform(
                lambda x: x.fillna(x.mean())
            )
            result[col] = result[col].fillna(circuit_means)

        # Level 3: Fill from global mean
        global_mean = result[col].mean()
        if np.isfinite(global_mean):
            result[col] = result[col].fillna(global_mean)
        else:
            result[col] = result[col].fillna(0.0)

    # Special handling for compound deg rates: cross-compound fallback
    _fill_compound_deg_rates(result)

    # Compute expected FP3-to-race delta from historical data at this circuit.
    # FP3 is low-fuel qualifying sims; race pace is ~3s slower due to fuel,
    # tyre management, traffic.  The delta is circuit-specific and predictable.
    #
    # IMPORTANT: Use leave-one-out to avoid data leakage in LORO CV.
    # For each race, the delta is the circuit average from all OTHER races
    # at the same circuit — never including the race itself.
    if (
        "circuit_baseline_pace" in result.columns
        and "lap_time_seconds" in result.columns
        and "circuit_id" in result.columns
        and "race_id" in result.columns
    ):
        # Compute actual FP3-to-race delta per race (one value per race_id)
        race_summary = result.groupby("race_id").agg(
            race_median=("lap_time_seconds", "median"),
            baseline=("circuit_baseline_pace", "first"),
            circuit=("circuit_id", "first"),
        )
        race_summary["actual_delta"] = race_summary["race_median"] - race_summary["baseline"]
        global_delta = race_summary["actual_delta"].mean()

        # Leave-one-out: for each race, average delta from OTHER races at same circuit
        loo_deltas = {}
        for race_id, row in race_summary.iterrows():
            same_circuit = race_summary[
                (race_summary["circuit"] == row["circuit"]) & (race_summary.index != race_id)
            ]
            if not same_circuit.empty:
                loo_deltas[race_id] = same_circuit["actual_delta"].mean()
            else:
                # Only race at this circuit — fall back to global mean
                loo_deltas[race_id] = global_delta if np.isfinite(global_delta) else 3.0

        # Map back to all laps
        result["expected_fp3_race_delta"] = result["race_id"].map(loo_deltas)

        # Fill any remaining NaN
        fill_val = global_delta if np.isfinite(global_delta) else 3.0
        result["expected_fp3_race_delta"] = result["expected_fp3_race_delta"].fillna(fill_val)
    elif "expected_fp3_race_delta" not in result.columns:
        result["expected_fp3_race_delta"] = 3.0  # Default ~3s fuel effect

    return result


def _fill_compound_deg_rates(df: pd.DataFrame) -> None:
    """Fill missing compound deg rates using other compounds as reference."""
    deg_cols = {
        "SOFT": "fp_deg_rate_soft",
        "MEDIUM": "fp_deg_rate_medium",
        "HARD": "fp_deg_rate_hard",
    }

    for compound, col in deg_cols.items():
        if col not in df.columns:
            continue

        mask = df[col].isna()
        if not mask.any():
            continue

        ratio = _COMPOUND_DEG_RATIO[compound]
        # Try to fill from other compounds
        for other_compound, other_col in deg_cols.items():
            if other_col == col or other_col not in df.columns:
                continue
            other_ratio = _COMPOUND_DEG_RATIO[other_compound]
            scale = ratio / other_ratio
            still_missing = df[col].isna()
            df.loc[still_missing, col] = df.loc[still_missing, other_col] * scale


def build_weekend_calibration(raw_dir: Path, race_ids: list[str]) -> pd.DataFrame:
    """Build weekend calibration features for a list of races.

    Args:
        raw_dir: Directory containing raw parquet files.
        race_ids: List of race identifiers ("{year}_{round:02d}").

    Returns:
        DataFrame with columns: race_id, driver_id, + weekend calibration features.
    """
    all_results = []

    for race_id in race_ids:
        practice_df = load_practice_laps(raw_dir, race_id)
        qualifying_df = load_qualifying_laps(raw_dir, race_id)

        if practice_df.empty and qualifying_df.empty:
            logger.debug(f"No practice/qualifying data for {race_id}")
            continue

        weekend = compute_weekend_features(practice_df, qualifying_df)
        if not weekend.empty:
            all_results.append(weekend)

    if not all_results:
        logger.info("No weekend calibration data found for any race")
        return pd.DataFrame(columns=["race_id", "driver_id", *WEEKEND_FEATURE_COLS])

    result = pd.concat(all_results, ignore_index=True)
    logger.info(
        f"Built weekend calibration: {len(result)} driver-race entries "
        f"across {result['race_id'].nunique()} races"
    )
    return result
