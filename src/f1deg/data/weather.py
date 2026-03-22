"""Open-Meteo weather API client for supplemental weather data.

Used as a fallback when FastF1 weather data has gaps.
"""

from datetime import date
import logging

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(
    lat: float,
    lon: float,
    race_date: date,
    hourly_vars: list[str] | None = None,
) -> dict:
    """Fetch historical weather data from Open-Meteo for a given location and date.

    Args:
        lat: Latitude of the circuit.
        lon: Longitude of the circuit.
        race_date: Date of the race.
        hourly_vars: List of hourly weather variables to fetch.

    Returns:
        Dict with hourly weather data.
    """
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ]

    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": race_date.isoformat(),
        "end_date": race_date.isoformat(),
        "hourly": ",".join(hourly_vars),
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    data: dict = response.json()

    logger.debug(f"Fetched weather for ({lat}, {lon}) on {race_date}")
    result: dict = data.get("hourly", {})
    return result


def get_race_weather(
    circuit_coords: dict[str, float],
    race_date: date,
    race_hour_utc: int = 14,
) -> dict[str, float]:
    """Get weather conditions at approximate race time.

    Args:
        circuit_coords: Dict with 'lat' and 'lon' keys.
        race_date: Date of the race.
        race_hour_utc: Approximate race start hour in UTC.

    Returns:
        Dict with weather values at race time.
    """
    hourly = fetch_weather(
        lat=circuit_coords["lat"],
        lon=circuit_coords["lon"],
        race_date=race_date,
    )

    if not hourly or "time" not in hourly:
        logger.warning("No hourly data returned from Open-Meteo")
        return {}

    # Find the index closest to race hour
    times = hourly["time"]
    target = f"{race_date.isoformat()}T{race_hour_utc:02d}:00"
    try:
        idx = times.index(target)
    except ValueError:
        idx = min(race_hour_utc, len(times) - 1)

    result = {}
    key_map = {
        "temperature_2m": "air_temp",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "precipitation": "rainfall",
    }

    for api_key, output_key in key_map.items():
        if api_key in hourly and idx < len(hourly[api_key]):
            result[output_key] = hourly[api_key][idx]

    return result
