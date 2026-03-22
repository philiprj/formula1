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


def load_session(year: int, round_num: int) -> fastf1.core.Session:
    """Load a race session with laps and weather data."""
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, weather=True, telemetry=False, messages=False)
    return session


def _load_session_with_retry(
    year: int,
    round_num: int,
    wait_minutes: int = 30,
    max_retries: int = 3,
) -> fastf1.core.Session:
    """Load a session, waiting and retrying if the rate limit is hit."""
    for attempt in range(max_retries):
        try:
            return load_session(year, round_num)
        except RateLimitExceededError:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Rate limit hit for {year} R{round_num}. "
                    f"Waiting {wait_minutes} minutes before retrying..."
                )
                time.sleep(wait_minutes * 60)
            else:
                raise


def extract_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """Extract lap data from a session, merging weather information."""
    laps = session.laps.copy()

    if laps.empty:
        logger.warning(f"No laps found for {session.event['EventName']} {session.event.year}")
        return pd.DataFrame()

    # Merge weather data per lap (get_weather_data returns weather-only columns,
    # so we join them back onto the laps by index)
    weather = session.weather_data
    if weather is not None and not weather.empty:
        weather_per_lap = laps.get_weather_data()
        # Avoid duplicate 'Time' column from weather
        weather_cols = [c for c in weather_per_lap.columns if c != "Time"]
        laps = laps.join(weather_per_lap[weather_cols])

    # Add session metadata
    laps["Year"] = session.event.year
    laps["RoundNumber"] = session.event["RoundNumber"]
    laps["CircuitKey"] = session.event["EventName"]
    laps["TotalLaps"] = session.total_laps

    return laps


def ingest_season(year: int, output_dir: Path) -> list[Path]:
    """Ingest all race sessions for a given season.

    Returns list of output Parquet file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = fastf1.get_event_schedule(year)
    output_files = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        if round_num == 0:  # Skip testing events
            continue

        event_name = event["EventName"].replace(" ", "_").lower()
        output_path = output_dir / f"{year}_{round_num:02d}_{event_name}.parquet"

        if output_path.exists():
            logger.info(f"Skipping {year} R{round_num} {event_name} (already exists)")
            output_files.append(output_path)
            continue

        logger.info(f"Loading {year} R{round_num} {event_name}...")
        try:
            session = _load_session_with_retry(year, round_num)
            laps = extract_laps(session)

            if laps.empty:
                logger.warning(f"No laps extracted for {year} R{round_num}")
                continue

            # Convert timedelta columns to seconds for Parquet compatibility
            timedelta_cols = laps.select_dtypes(include=["timedelta64"]).columns
            for col in timedelta_cols:
                laps[f"{col}_seconds"] = laps[col].dt.total_seconds()

            laps.to_parquet(output_path, index=False)
            output_files.append(output_path)
            logger.info(f"Saved {len(laps)} laps to {output_path}")

        except Exception as e:
            logger.error(f"Failed to load {year} R{round_num}: {e}")
            continue

    return output_files


def ingest_all(config: dict | None = None) -> list[Path]:
    """Ingest all configured seasons."""
    if config is None:
        config = load_config()

    enable_cache(config.get("cache_dir"))
    output_dir = Path(config["data_dir"]) / "raw"
    seasons = config.get("seasons", [2022, 2023, 2024, 2025])

    all_files = []
    for year in seasons:
        logger.info(f"--- Ingesting {year} season ---")
        files = ingest_season(year, output_dir)
        all_files.extend(files)
        logger.info(f"Completed {year}: {len(files)} races")

    logger.info(f"Total: {len(all_files)} race files ingested")
    return all_files
