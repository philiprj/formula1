"""Jolpica (Ergast replacement) API client for race results and retirement data."""

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Statuses that indicate a driver did NOT finish normally
RETIREMENT_STATUSES = {
    "Accident",
    "Collision",
    "Engine",
    "Gearbox",
    "Hydraulics",
    "Brakes",
    "Suspension",
    "Transmission",
    "Electrical",
    "Puncture",
    "Overheating",
    "Mechanical",
    "Tyre",
    "Wheel",
    "Power Unit",
    "ERS",
    "Retired",
    "Spun off",
    "Withdrew",
    "Disqualified",
    "Excluded",
    "Did not finish",
    "Oil leak",
    "Water leak",
    "Fuel pressure",
    "Throttle",
    "Steering",
    "Technical",
    "Electronics",
    "Clutch",
    "Differential",
    "Fuel system",
    "Fuel pump",
    "Track rod",
    "Driveshaft",
    "Exhaust",
    "Turbo",
    "Cooling",
    "Battery",
    "Vibrations",
    "Water pressure",
    "Oil pressure",
    "Fire",
    "Damage",
    "Debris",
}

# Statuses indicating a lapped finish (not a retirement)
LAPPED_PATTERN = r"^\+\d+ Lap"


def fetch_race_results(year: int, round_num: int) -> list[dict]:
    """Fetch race results for a specific race from Jolpica API.

    Returns list of result records with driver status and finishing info.
    """
    url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/results.json"
    all_results = []
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

        results = races[0].get("Results", [])
        if not results:
            break

        all_results.extend(results)

        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break

        time.sleep(0.5)

    return all_results


def is_retirement(status: str) -> bool:
    """Determine if a result status indicates a retirement/DNF."""
    if not status:
        return False
    if status == "Finished":
        return False
    # Lapped finishes are not retirements
    if pd.Series([status]).str.match(LAPPED_PATTERN).iloc[0]:
        return False
    # Check known retirement statuses (case-insensitive partial match)
    status_lower = status.lower()
    for ret_status in RETIREMENT_STATUSES:
        if ret_status.lower() in status_lower or status_lower in ret_status.lower():
            return True
    # If status is not "Finished" and not lapped, likely a retirement
    return True


def get_season_results(year: int, num_rounds: int) -> pd.DataFrame:
    """Fetch race results for all races in a season.

    Args:
        year: Season year.
        num_rounds: Number of rounds in the season.

    Returns:
        DataFrame with columns: year, round, driver_id, position_final,
        grid_position, laps_completed, status, did_retire.
    """
    records = []

    for round_num in range(1, num_rounds + 1):
        logger.info(f"Fetching results: {year} R{round_num}")
        try:
            results = fetch_race_results(year, round_num)
            for result in results:
                status = result.get("status", "")
                laps = int(result.get("laps", 0))
                grid = int(result.get("grid", 0))
                position = result.get("position", "")

                records.append(
                    {
                        "year": year,
                        "round": round_num,
                        "driver_id": result.get("Driver", {}).get("code", ""),
                        "position_final": int(position) if position.isdigit() else None,
                        "grid_position": grid,
                        "laps_completed": laps,
                        "status": status,
                        "did_retire": is_retirement(status),
                    }
                )
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed to fetch results for {year} R{round_num}: {e}")
            continue

    df = pd.DataFrame(records)
    logger.info(
        f"Fetched {len(df)} results for {year} "
        f"({df['did_retire'].sum() if len(df) > 0 else 0} retirements)"
    )
    return df
