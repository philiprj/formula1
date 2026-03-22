"""Jolpica (Ergast replacement) API client for pit stop data."""

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"


def fetch_pit_stops(year: int, round_num: int) -> list[dict]:
    """Fetch pit stop data for a specific race from Jolpica API.

    Returns list of pit stop records.
    """
    url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/pitstops.json"
    all_stops = []
    offset = 0
    limit = 100

    while True:
        params = {"limit": limit, "offset": offset}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        race_table = data.get("MRData", {}).get("RaceTable", {})
        races = race_table.get("Races", [])

        if not races:
            break

        stops = races[0].get("PitStops", [])
        if not stops:
            break

        all_stops.extend(stops)

        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break

        time.sleep(0.5)  # Rate limit: 200 req/hour

    return all_stops


def get_season_pit_stops(year: int, num_rounds: int) -> pd.DataFrame:
    """Fetch pit stops for all races in a season.

    Args:
        year: Season year.
        num_rounds: Number of rounds in the season.

    Returns:
        DataFrame with columns: year, round, driver_id, stop, lap, time, duration.
    """
    records = []

    for round_num in range(1, num_rounds + 1):
        logger.info(f"Fetching pit stops: {year} R{round_num}")
        try:
            stops = fetch_pit_stops(year, round_num)
            for stop in stops:
                records.append(
                    {
                        "year": year,
                        "round": round_num,
                        "driver_id": stop.get("driverId", ""),
                        "stop_number": int(stop.get("stop", 0)),
                        "lap": int(stop.get("lap", 0)),
                        "time_of_day": stop.get("time", ""),
                        "duration_seconds": _parse_duration(stop.get("duration", "")),
                    }
                )
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.error(f"Failed to fetch pit stops for {year} R{round_num}: {e}")
            continue

    df = pd.DataFrame(records)
    logger.info(f"Fetched {len(df)} pit stops for {year}")
    return df


def _parse_duration(duration_str: str) -> float | None:
    """Parse pit stop duration string (e.g., '23.456') to float seconds."""
    if not duration_str:
        return None
    try:
        parts = duration_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(duration_str)
    except (ValueError, TypeError):
        return None
