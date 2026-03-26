"""Lap data filtering functions.

Each filter is a pure function: (DataFrame) -> DataFrame.
Filters compose via apply_filters().

Flagging functions add boolean columns (is_outlier, etc.) without removing rows.
"""

from collections.abc import Callable
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_accurate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only laps marked as accurate by FastF1."""
    if "IsAccurate" not in df.columns:
        logger.warning("IsAccurate column not found, skipping filter")
        return df
    before = len(df)
    result = df[df["IsAccurate"] == True].copy()  # noqa: E712
    logger.debug(f"filter_accurate: {before} -> {len(result)} laps")
    return result


def filter_track_status(
    df: pd.DataFrame,
    exclude_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Remove laps affected by safety car, VSC, or red flag.

    TrackStatus is a string of status digit codes.
    Codes: 1=Green, 2=Yellow, 4=SC, 5=Red, 6=VSC, 7=VSC Ending.
    """
    if "TrackStatus" not in df.columns:
        logger.warning("TrackStatus column not found, skipping filter")
        return df

    if exclude_codes is None:
        exclude_codes = ["4", "5", "6"]

    before = len(df)

    def has_excluded_status(status: str) -> bool:
        if pd.isna(status):
            return False
        return any(code in str(status) for code in exclude_codes)

    mask = ~df["TrackStatus"].apply(has_excluded_status)
    result = df[mask].copy()
    logger.debug(f"filter_track_status: {before} -> {len(result)} laps")
    return result


def flag_track_status(
    df: pd.DataFrame,
    sc_codes: list[str] | None = None,
    vsc_codes: list[str] | None = None,
    red_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Flag SC/VSC/red flag laps instead of removing them.

    Adds boolean columns: is_under_sc, is_under_vsc, is_under_red_flag.
    This preserves SC laps in the training data so models can learn
    SC-period behavior (critical for pit strategy decisions).

    Laps under red flag are still filtered out (race is stopped).
    """
    if "TrackStatus" not in df.columns:
        logger.warning("TrackStatus column not found, skipping track status flagging")
        return df

    if sc_codes is None:
        sc_codes = ["4"]
    if vsc_codes is None:
        vsc_codes = ["6", "7"]
    if red_codes is None:
        red_codes = ["5"]

    result = df.copy()

    def _has_code(status, codes):
        if pd.isna(status):
            return False
        return any(c in str(status) for c in codes)

    result["is_under_sc"] = result["TrackStatus"].apply(lambda s: _has_code(s, sc_codes))
    result["is_under_vsc"] = result["TrackStatus"].apply(lambda s: _has_code(s, vsc_codes))
    result["is_under_red_flag"] = result["TrackStatus"].apply(lambda s: _has_code(s, red_codes))

    # Still remove red flag laps (race is stopped, times meaningless)
    before = len(result)
    result = result[~result["is_under_red_flag"]].copy()

    sc_count = result["is_under_sc"].sum()
    vsc_count = result["is_under_vsc"].sum()
    logger.debug(
        f"flag_track_status: {before} -> {len(result)} laps "
        f"(kept {sc_count} SC + {vsc_count} VSC laps, removed {before - len(result)} red flag laps)"
    )
    return result


def filter_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pit-in and pit-out laps (inflated times)."""
    before = len(df)

    pit_in_mask = pd.Series(True, index=df.index)
    pit_out_mask = pd.Series(True, index=df.index)

    if "PitInTime" in df.columns:
        pit_in_mask = df["PitInTime"].isna()
    if "PitOutTime" in df.columns:
        pit_out_mask = df["PitOutTime"].isna()

    result = df[pit_in_mask & pit_out_mask].copy()
    logger.debug(f"filter_pit_laps: {before} -> {len(result)} laps")
    return result


def filter_first_lap(df: pd.DataFrame) -> pd.DataFrame:
    """Remove lap 1 from each driver (grid start chaos)."""
    if "LapNumber" not in df.columns:
        logger.warning("LapNumber column not found, skipping filter")
        return df
    before = len(df)
    result = df[df["LapNumber"] > 1].copy()
    logger.debug(f"filter_first_lap: {before} -> {len(result)} laps")
    return result


def filter_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = 3.0,
    group_col: str = "race_id",
) -> pd.DataFrame:
    """Remove laps with times exceeding median + multiplier*IQR per group.

    Falls back to global filtering if group_col is not present.
    """
    time_col = "lap_time_seconds" if "lap_time_seconds" in df.columns else "LapTime_seconds"
    if time_col not in df.columns:
        logger.warning(f"{time_col} column not found, skipping outlier filter")
        return df

    before = len(df)

    if group_col in df.columns:
        mask = pd.Series(True, index=df.index)
        for _, group in df.groupby(group_col):
            q1 = group[time_col].quantile(0.25)
            q3 = group[time_col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + iqr_multiplier * iqr
            group_mask = group[time_col] <= upper_bound
            mask.loc[group.index] = group_mask
        result = df[mask].copy()
    else:
        q1 = df[time_col].quantile(0.25)
        q3 = df[time_col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr_multiplier * iqr
        result = df[df[time_col] <= upper_bound].copy()

    logger.debug(f"filter_outliers: {before} -> {len(result)} laps")
    return result


def flag_outliers_compound_aware(
    df: pd.DataFrame,
    zscore_threshold: float = 2.5,
    iqr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Flag outlier laps using compound-aware z-scores.

    Groups by (race_id, compound) to account for natural pace differences
    between tire compounds. Flags rather than removes — adds 'is_outlier'
    and 'outlier_reason' columns.

    Uses z-score as primary method. Falls back to IQR for small groups
    (< 10 laps) where z-score is unreliable.
    """
    time_col = "lap_time_seconds" if "lap_time_seconds" in df.columns else "LapTime_seconds"
    if time_col not in df.columns:
        logger.warning(f"{time_col} column not found, skipping compound outlier flagging")
        return df

    result = df.copy()
    if "is_outlier" not in result.columns:
        result["is_outlier"] = False
    if "outlier_reason" not in result.columns:
        result["outlier_reason"] = pd.Series("", index=result.index, dtype="object")

    group_cols = []
    if "race_id" in result.columns:
        group_cols.append("race_id")
    if "compound" in result.columns:
        group_cols.append("compound")

    if not group_cols:
        group_cols = ["race_id"] if "race_id" in result.columns else []

    if not group_cols:
        # Fall back to global z-score
        mean = result[time_col].mean()
        std = result[time_col].std()
        if std > 0:
            zscore = (result[time_col] - mean) / std
            mask = zscore > zscore_threshold
            result.loc[mask, "is_outlier"] = True
            result.loc[mask, "outlier_reason"] = "compound_zscore"
        return result

    # Pre-compute per-race condition flags for adaptive thresholds
    race_is_wet = {}
    race_has_sc = {}
    if "race_id" in result.columns:
        if "rainfall" in result.columns:
            wet_pct = result.groupby("race_id")["rainfall"].apply(
                lambda x: (x.fillna(0).astype(float) > 0).mean()
            )
            race_is_wet = (wet_pct > 0.1).to_dict()
        if "had_sc_this_stint" in result.columns:
            sc_pct = result.groupby("race_id")["had_sc_this_stint"].apply(
                lambda x: x.astype(float).mean()
            )
            race_has_sc = (sc_pct > 0.2).to_dict()

    for group_key, group in result.groupby(group_cols):
        times = group[time_col]
        if len(times) < 3:
            continue

        # Adaptive threshold: widen for rain/SC races
        effective_threshold = zscore_threshold
        effective_iqr = iqr_multiplier
        race_id = group_key[0] if isinstance(group_key, tuple) else group_key
        if race_is_wet.get(race_id, False):
            effective_threshold = max(zscore_threshold, 3.5)
            effective_iqr = max(iqr_multiplier, 2.5)
        elif race_has_sc.get(race_id, False):
            effective_threshold = max(zscore_threshold, 3.0)
            effective_iqr = max(iqr_multiplier, 2.25)

        if len(times) < 10:
            # Small group: use IQR
            q1, q3 = times.quantile(0.25), times.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + effective_iqr * iqr
            mask = times > upper
        else:
            # Large group: use z-score
            mean = times.mean()
            std = times.std()
            if std == 0:
                continue
            zscore = (times - mean) / std
            mask = zscore > effective_threshold

        flagged_idx = group.index[mask]
        result.loc[flagged_idx, "is_outlier"] = True
        result.loc[flagged_idx, "outlier_reason"] = "compound_zscore"

    flagged = result["is_outlier"].sum()
    logger.debug(f"flag_outliers_compound_aware: flagged {flagged}/{len(result)} laps")
    return result


def flag_yellow_adjacent(
    df: pd.DataFrame,
    adjacent_laps: int = 1,
    sc_adjacent_laps: int = 2,
) -> pd.DataFrame:
    """Flag laps adjacent to yellow flag or safety car periods.

    Drivers lift when they see yellows ahead, affecting lap times even
    if their own TrackStatus is still green. Flags the laps before/after
    yellow flag periods for each driver.

    Args:
        df: Lap DataFrame with TrackStatus, driver_id/Driver, and lap_number/LapNumber.
        adjacent_laps: Number of laps to flag around yellow flags.
        sc_adjacent_laps: Number of laps to flag before SC deployment.
    """
    if "TrackStatus" not in df.columns:
        logger.warning("TrackStatus column not found, skipping yellow adjacency flagging")
        return df

    result = df.copy()
    if "is_outlier" not in result.columns:
        result["is_outlier"] = False
    if "outlier_reason" not in result.columns:
        result["outlier_reason"] = pd.Series("", index=result.index, dtype="object")

    driver_col = "driver_id" if "driver_id" in result.columns else "Driver"
    lap_col = "lap_number" if "lap_number" in result.columns else "LapNumber"
    race_col = "race_id" if "race_id" in result.columns else None

    if driver_col not in result.columns or lap_col not in result.columns:
        logger.warning("Required columns missing for yellow adjacency flagging")
        return result

    group_cols = [driver_col]
    if race_col and race_col in result.columns:
        group_cols = [race_col, driver_col]

    for _, group in result.groupby(group_cols):
        sorted_group = group.sort_values(lap_col)
        statuses = sorted_group["TrackStatus"].astype(str)

        # Find yellow flag laps (code "2")
        yellow_mask = statuses.apply(lambda s: "2" in str(s) if pd.notna(s) else False)
        # Find SC deployment laps (code "4")
        sc_mask = statuses.apply(lambda s: "4" in str(s) if pd.notna(s) else False)

        yellow_laps = sorted_group.loc[yellow_mask, lap_col].values
        sc_laps = sorted_group.loc[sc_mask, lap_col].values

        affected_laps = set()

        for yl in yellow_laps:
            for offset in range(-adjacent_laps, adjacent_laps + 1):
                affected_laps.add(yl + offset)

        for sl in sc_laps:
            for offset in range(-sc_adjacent_laps, 1):  # before SC only
                affected_laps.add(sl + offset)

        # Flag laps that are adjacent but NOT themselves yellow/SC
        for idx, row in sorted_group.iterrows():
            ln = row[lap_col]
            status = str(row["TrackStatus"]) if pd.notna(row["TrackStatus"]) else ""
            is_already_flagged = any(c in status for c in ["2", "4", "5", "6"])
            if ln in affected_laps and not is_already_flagged:
                result.loc[idx, "is_outlier"] = True
                existing = result.loc[idx, "outlier_reason"]
                reason = "yellow_adjacent"
                if existing and existing != reason:
                    reason = f"{existing},{reason}"
                result.loc[idx, "outlier_reason"] = reason

    flagged = result["outlier_reason"].str.contains("yellow_adjacent", na=False).sum()
    logger.debug(f"flag_yellow_adjacent: flagged {flagged} additional laps")
    return result


# Registry of available filters
FILTER_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "accurate": filter_accurate,
    "track_status": filter_track_status,
    "track_status_flag": flag_track_status,
    "pit_laps": filter_pit_laps,
    "first_lap": filter_first_lap,
    "outliers": filter_outliers,
    "outliers_compound": flag_outliers_compound_aware,
    "yellow_adjacent": flag_yellow_adjacent,
}


def apply_filters(
    df: pd.DataFrame,
    filter_names: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Apply a sequence of filters to the DataFrame.

    Args:
        df: Input lap data.
        filter_names: List of filter names from FILTER_REGISTRY.
            If None, applies all filters in default order.
        config: Optional config dict for filter parameters.
    """
    if filter_names is None:
        filter_names = ["accurate", "track_status", "pit_laps", "first_lap", "outliers"]

    before = len(df)
    result = df

    for name in filter_names:
        if name not in FILTER_REGISTRY:
            logger.warning(f"Unknown filter: {name}, skipping")
            continue

        fn = FILTER_REGISTRY[name]

        # Pass config-based kwargs for filters that accept them
        kwargs = {}
        if name == "track_status" and config:
            exclude = config.get("features", {}).get("track_status_exclude")
            if exclude:
                kwargs["exclude_codes"] = exclude
        elif name == "outliers" and config:
            multiplier = config.get("features", {}).get("outlier_iqr_multiplier")
            if multiplier:
                kwargs["iqr_multiplier"] = multiplier

        result = fn(result, **kwargs)

    logger.info(
        f"Filtering complete: {before} -> {len(result)} laps ({before - len(result)} removed)"
    )
    return result
