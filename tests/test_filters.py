"""Tests for lap data filtering functions."""

import pandas as pd

from f1deg.data.filters import (
    apply_filters,
    filter_accurate,
    filter_first_lap,
    filter_outliers,
    filter_pit_laps,
    filter_track_status,
)


def test_filter_accurate(sample_raw_laps):
    result = filter_accurate(sample_raw_laps)
    assert result["IsAccurate"].all()
    assert len(result) < len(sample_raw_laps)


def test_filter_accurate_missing_column():
    df = pd.DataFrame({"LapNumber": [1, 2, 3]})
    result = filter_accurate(df)
    assert len(result) == 3  # No filtering if column missing


def test_filter_track_status(sample_raw_laps):
    result = filter_track_status(sample_raw_laps)
    # No laps with SC code "4" should remain
    assert not any(result["TrackStatus"].str.contains("4", na=False))


def test_filter_track_status_custom_codes():
    df = pd.DataFrame(
        {
            "TrackStatus": ["1", "4", "6", "1", "5"],
        }
    )
    result = filter_track_status(df, exclude_codes=["4"])
    assert len(result) == 4  # Only code "4" removed, rows with "1", "6", "1", "5" remain


def test_filter_pit_laps(sample_raw_laps):
    result = filter_pit_laps(sample_raw_laps)
    assert all(result["PitInTime"].isna())
    assert all(result["PitOutTime"].isna())


def test_filter_first_lap(sample_raw_laps):
    result = filter_first_lap(sample_raw_laps)
    assert all(result["LapNumber"] > 1)


def test_filter_outliers(sample_processed_laps):
    # Inject an extreme outlier
    df = sample_processed_laps.copy()
    df.loc[0, "lap_time_seconds"] = 200.0  # Way too slow
    result = filter_outliers(df, iqr_multiplier=3.0, group_col="race_id")
    assert len(result) < len(df)


def test_apply_filters(sample_raw_laps):
    # Need lap_time_seconds for outlier filter
    df = sample_raw_laps.copy()
    df["lap_time_seconds"] = df["LapTime_seconds"]
    result = apply_filters(df, filter_names=["accurate", "track_status", "pit_laps", "first_lap"])
    assert len(result) < len(df)
    assert result["IsAccurate"].all()
