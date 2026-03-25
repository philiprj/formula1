"""FastF1 data ingestion pipeline.

Loads race sessions from FastF1, extracts lap and weather data,
and saves as Parquet files for downstream processing.
"""

import logging
from pathlib import Path
import time

import fastf1
from fastf1.exceptions import RateLimitExceededError
import pandas as pd

from f1deg.config import load_config

logger = logging.getLogger(__name__)


def enable_cache(cache_dir: str | None = None) -> None:
    """Enable FastF1 caching. Falls back to data/cache in the project root."""
    from f1deg.config import PROJECT_ROOT

    if not cache_dir:
        cache_dir = str(PROJECT_ROOT / "data" / "cache")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def load_session(year: int, round_num: int, session_type: str = "R") -> fastf1.core.Session:
    """Load a session with laps and weather data.

    Args:
        year: Season year.
        round_num: Round number.
        session_type: One of "FP1", "FP2", "FP3", "Q", "S", "R".
    """
    session = fastf1.get_session(year, round_num, session_type)
    session.load(laps=True, weather=True, telemetry=False, messages=False)
    return session


def _load_session_with_retry(
    year: int,
    round_num: int,
    session_type: str = "R",
    wait_minutes: int = 30,
    max_retries: int = 3,
) -> fastf1.core.Session:
    """Load a session, waiting and retrying if the rate limit is hit."""
    for attempt in range(max_retries):
        try:
            return load_session(year, round_num, session_type)
        except RateLimitExceededError:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Rate limit hit for {year} R{round_num} {session_type}. "
                    f"Waiting {wait_minutes} minutes before retrying..."
                )
                time.sleep(wait_minutes * 60)
            else:
                raise


def compute_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """Compute time gaps between cars at each lap boundary.

    Uses cumulative session Time to calculate the gap to the car ahead
    and behind for each driver on each lap. No telemetry needed.

    Returns DataFrame with columns: Driver, LapNumber, gap_ahead_seconds, gap_behind_seconds.
    """
    if "Time" not in laps.columns or "LapNumber" not in laps.columns:
        logger.warning("Time or LapNumber column missing, skipping gap computation")
        return pd.DataFrame(
            columns=["Driver", "LapNumber", "gap_ahead_seconds", "gap_behind_seconds"]
        )

    records = []
    for lap_num, lap_group in laps.groupby("LapNumber"):
        # Filter to laps with valid Time
        valid = lap_group.dropna(subset=["Time"]).copy()
        if len(valid) < 2:
            continue

        # Convert Time to total seconds for comparison
        if pd.api.types.is_timedelta64_dtype(valid["Time"]):
            valid["_time_s"] = valid["Time"].dt.total_seconds()
        else:
            valid["_time_s"] = valid["Time"]

        # Sort by session time (crossing order)
        valid = valid.sort_values("_time_s")
        times = valid["_time_s"].values
        drivers = valid["Driver"].values

        for i, (driver, _t) in enumerate(zip(drivers, times, strict=False)):
            gap_ahead = times[i] - times[i - 1] if i > 0 else None
            gap_behind = times[i + 1] - times[i] if i < len(times) - 1 else None
            records.append(
                {
                    "Driver": driver,
                    "LapNumber": lap_num,
                    "gap_ahead_seconds": gap_ahead,
                    "gap_behind_seconds": gap_behind,
                }
            )

    return pd.DataFrame(records)


def extract_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """Extract lap data from a session, merging weather and gap information."""
    laps = session.laps.copy()

    if laps.empty:
        logger.warning(f"No laps found for {session.event['EventName']} {session.event.year}")
        return pd.DataFrame()

    # Merge weather data per lap using merge_asof on LapStartTime.
    # FastF1's get_weather_data() has a non-unique index that creates
    # duplicate rows and leaves most laps without weather.  Instead,
    # we directly match each lap to the nearest weather reading by time.
    weather = session.weather_data
    if weather is not None and not weather.empty and "LapStartTime" in laps.columns:
        weather_cols = [
            c
            for c in [
                "AirTemp",
                "TrackTemp",
                "Humidity",
                "Pressure",
                "Rainfall",
                "WindSpeed",
                "WindDirection",
            ]
            if c in weather.columns
        ]
        wd = weather[["Time", *weather_cols]].sort_values("Time").copy()
        laps_sorted = laps.sort_values("LapStartTime")
        merged_weather = pd.merge_asof(
            laps_sorted[["LapStartTime"]].reset_index(),
            wd.rename(columns={"Time": "LapStartTime"}),
            on="LapStartTime",
            direction="nearest",
        ).set_index("index")[weather_cols]
        for col in weather_cols:
            laps[col] = merged_weather[col]

    # Compute inter-car gaps from lap crossing times
    gaps = compute_gaps(laps)
    if not gaps.empty:
        laps = laps.merge(gaps, on=["Driver", "LapNumber"], how="left")

    # Add session metadata
    laps["Year"] = session.event.year
    laps["RoundNumber"] = session.event["RoundNumber"]
    laps["CircuitKey"] = session.event["EventName"]
    laps["TotalLaps"] = session.total_laps
    laps["SessionType"] = session.name  # e.g. "Race", "Qualifying", "Practice 1"

    return laps


def _file_suffix_for_session(session_type: str) -> str:
    """Map session type code to filename suffix."""
    return {"FP1": "_fp1", "FP2": "_fp2", "FP3": "_fp3", "Q": "_q", "S": "_sprint"}.get(
        session_type, ""
    )


def ingest_season(
    year: int,
    output_dir: Path,
    session_types: list[str] | None = None,
) -> list[Path]:
    """Ingest sessions for a given season.

    Args:
        year: Season year.
        output_dir: Directory for output Parquet files.
        session_types: Session types to ingest (default: ["R"]).

    Returns list of output Parquet file paths.
    """
    if session_types is None:
        session_types = ["R"]

    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = fastf1.get_event_schedule(year)
    output_files = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        if round_num == 0:  # Skip testing events
            continue

        event_name = event["EventName"].replace(" ", "_").lower()

        for stype in session_types:
            suffix = _file_suffix_for_session(stype)
            output_path = output_dir / f"{year}_{round_num:02d}_{event_name}{suffix}.parquet"

            if output_path.exists():
                logger.info(f"Skipping {year} R{round_num} {stype} {event_name} (already exists)")
                output_files.append(output_path)
                continue

            logger.info(f"Loading {year} R{round_num} {stype} {event_name}...")
            try:
                session = _load_session_with_retry(year, round_num, stype)
                laps = extract_laps(session)

                if laps.empty:
                    logger.warning(f"No laps extracted for {year} R{round_num} {stype}")
                    continue

                # Convert timedelta columns to seconds for Parquet compatibility
                timedelta_cols = laps.select_dtypes(include=["timedelta64"]).columns
                for col in timedelta_cols:
                    laps[f"{col}_seconds"] = laps[col].dt.total_seconds()

                laps.to_parquet(output_path, index=False)
                output_files.append(output_path)
                logger.info(f"Saved {len(laps)} laps to {output_path}")

            except Exception as e:
                logger.error(f"Failed to load {year} R{round_num} {stype}: {e}")
                continue

    return output_files


def ingest_all(
    config: dict | None = None,
    session_types: list[str] | None = None,
) -> list[Path]:
    """Ingest all configured seasons.

    Args:
        config: Configuration dict. Loaded from base.yaml if None.
        session_types: Override session types to ingest. If None, uses config
            value (default: ["R"] for backward compatibility).
    """
    if config is None:
        config = load_config()

    enable_cache(config.get("cache_dir"))
    output_dir = Path(config["data_dir"]) / "raw"
    seasons = config.get("seasons", [2022, 2023, 2024, 2025])

    if session_types is None:
        session_types = config.get("session_types", ["R"])

    all_files = []
    for year in seasons:
        logger.info(f"--- Ingesting {year} season ({', '.join(session_types)}) ---")
        files = ingest_season(year, output_dir, session_types=session_types)
        all_files.extend(files)
        logger.info(f"Completed {year}: {len(files)} files")

    logger.info(f"Total: {len(all_files)} session files ingested")
    return all_files
