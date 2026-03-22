"""Lap data filtering functions.

Each filter is a pure function: (DataFrame) -> DataFrame.
Filters compose via apply_filters().
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


# Registry of available filters
FILTER_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "accurate": filter_accurate,
    "track_status": filter_track_status,
    "pit_laps": filter_pit_laps,
    "first_lap": filter_first_lap,
    "outliers": filter_outliers,
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
