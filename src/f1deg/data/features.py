"""Feature engineering for tire degradation modeling.

Transforms raw FastF1 lap data into model-ready features.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1deg.config import load_config

logger = logging.getLogger(__name__)


_PRACTICE_SUFFIXES = {"_fp1", "_fp2", "_fp3", "_q", "_sprint"}

# TrackStatus codes from FastF1
_SC_CODES = {"4"}  # Safety Car
_VSC_CODES = {"6", "7"}  # Virtual Safety Car (+ ending)
_RED_CODES = {"5"}  # Red Flag
_CAUTION_CODES = _SC_CODES | _VSC_CODES | _RED_CODES

# Mapping from FastF1 event names to config circuit keys
_CIRCUIT_NAME_TO_KEY = {
    "Abu Dhabi Grand Prix": "yas_marina",
    "Australian Grand Prix": "albert_park",
    "Austrian Grand Prix": "spielberg",
    "Azerbaijan Grand Prix": "baku",
    "Bahrain Grand Prix": "bahrain",
    "Belgian Grand Prix": "spa",
    "British Grand Prix": "silverstone",
    "Canadian Grand Prix": "montreal",
    "Chinese Grand Prix": "shanghai",
    "Dutch Grand Prix": "zandvoort",
    "Emilia Romagna Grand Prix": "imola",
    "French Grand Prix": "paul_ricard",
    "Hungarian Grand Prix": "hungaroring",
    "Italian Grand Prix": "monza",
    "Japanese Grand Prix": "suzuka",
    "Las Vegas Grand Prix": "las_vegas",
    "Mexico City Grand Prix": "mexico",
    "Miami Grand Prix": "miami",
    "Monaco Grand Prix": "monaco",
    "Qatar Grand Prix": "lusail",
    "Saudi Arabian Grand Prix": "jeddah",
    "Singapore Grand Prix": "marina_bay",
    "Spanish Grand Prix": "barcelona",
    "São Paulo Grand Prix": "interlagos",
    "United States Grand Prix": "cota",
}

# Compound class mapping
_COMPOUND_CLASS = {
    "SOFT": "dry",
    "MEDIUM": "dry",
    "HARD": "dry",
    "INTERMEDIATE": "inter",
    "WET": "wet",
}


def compute_sc_rain_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute safety car and rain proximity features from RAW lap data.

    Must be called BEFORE track_status filtering removes SC/VSC/Red laps,
    since these features need the full TrackStatus sequence.

    Returns a DataFrame with columns:
        race_id, driver_id, lap_number,
        laps_since_sc_end, laps_since_red_flag, had_sc_this_stint,
        compound_class, is_wet_running, compound_class_changed_this_stint,
        laps_since_compound_class_change, sub_race_id
    """
    # Determine column names (raw FastF1 vs processed)
    driver_col = "Driver" if "Driver" in raw_df.columns else "driver_id"
    lap_col = "LapNumber" if "LapNumber" in raw_df.columns else "lap_number"

    # Build race_id if not present
    df = raw_df.copy()
    if "race_id" not in df.columns and "Year" in df.columns and "RoundNumber" in df.columns:
        df["race_id"] = df["Year"].astype(str) + "_" + df["RoundNumber"].astype(str).str.zfill(2)
    if "race_id" not in df.columns:
        return pd.DataFrame()

    # Normalize compound
    compound_col = "Compound" if "Compound" in df.columns else "compound"
    if compound_col in df.columns:
        df["_compound_upper"] = df[compound_col].astype(str).str.upper()
    else:
        df["_compound_upper"] = "UNKNOWN"

    results = []

    for (race_id, driver_id), group in df.groupby(["race_id", driver_col]):
        group = group.sort_values(lap_col)
        laps = group[lap_col].values
        statuses = (
            group["TrackStatus"].astype(str).values
            if "TrackStatus" in group.columns
            else ["1"] * len(group)
        )
        compounds = group["_compound_upper"].values

        # Track SC/VSC/Red flag state
        last_sc_end_lap = -999
        last_red_end_lap = -999
        in_sc = False
        in_red = False
        sc_in_current_stint = False
        sub_race = 1

        # Track compound class changes
        prev_compound_class = (
            _COMPOUND_CLASS.get(compounds[0], "dry") if len(compounds) > 0 else "dry"
        )
        last_class_change_lap = -999

        # Track stint boundaries (TyreLife resets)
        tyre_life_col = "TyreLife" if "TyreLife" in group.columns else "tyre_life"
        tyre_lives = group[tyre_life_col].values if tyre_life_col in group.columns else None

        for i, (lap_num, status, compound) in enumerate(
            zip(laps, statuses, compounds, strict=True)
        ):
            status_str = str(status) if pd.notna(status) else "1"

            # Detect SC/VSC state transitions
            is_sc_now = any(c in status_str for c in _SC_CODES | _VSC_CODES)
            is_red_now = any(c in status_str for c in _RED_CODES)

            if in_sc and not is_sc_now:
                last_sc_end_lap = lap_num  # SC just ended
            if in_red and not is_red_now:
                last_red_end_lap = lap_num  # Red flag just ended
                sub_race += 1

            in_sc = is_sc_now
            in_red = is_red_now

            if is_sc_now or is_red_now:
                sc_in_current_stint = True

            # Detect stint boundaries (TyreLife reset)
            if tyre_lives is not None and i > 0 and tyre_lives[i] < tyre_lives[i - 1]:
                sc_in_current_stint = False  # Reset for new stint

            # Red flag also resets stint
            if is_red_now:
                sc_in_current_stint = True  # Will be True for the red flag stint

            # Compound class tracking
            comp_class = _COMPOUND_CLASS.get(compound, "dry")
            if comp_class != prev_compound_class:
                last_class_change_lap = lap_num
                prev_compound_class = comp_class

            # Compute features
            laps_since_sc = min(lap_num - last_sc_end_lap, 99) if last_sc_end_lap > 0 else 99
            laps_since_red = min(lap_num - last_red_end_lap, 99) if last_red_end_lap > 0 else 99
            laps_since_cc = (
                min(lap_num - last_class_change_lap, 99) if last_class_change_lap > 0 else 99
            )

            results.append(
                {
                    "race_id": race_id,
                    "driver_id": driver_id,
                    "lap_number": int(lap_num),
                    "laps_since_sc_end": min(laps_since_sc, 5),  # Cap at 5
                    "laps_since_red_flag": min(laps_since_red, 5),
                    "had_sc_this_stint": sc_in_current_stint,
                    "compound_class": comp_class,
                    "is_wet_running": float(comp_class != "dry"),
                    "compound_class_changed_this_stint": float(last_class_change_lap > 0),
                    "laps_since_compound_class_change": min(laps_since_cc, 10),
                    "sub_race_id": sub_race,
                }
            )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    # Normalize driver_id column name
    if driver_col != "driver_id":
        result_df = result_df.rename(columns={"driver_id": "driver_id"})

    logger.info(
        f"Computed SC/rain features: {len(result_df)} laps, "
        f"{(result_df['laps_since_sc_end'] < 5).sum()} post-SC laps, "
        f"{(result_df['is_wet_running'] > 0).sum()} wet laps"
    )
    return result_df


def _is_race_file(path: Path) -> bool:
    """Return True if the parquet file is a race session (not practice/qualifying)."""
    stem = path.stem.lower()
    return not any(stem.endswith(s) for s in _PRACTICE_SUFFIXES)


def load_raw_laps(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate raw race Parquet files (excluding practice/qualifying)."""
    files = sorted(f for f in raw_dir.glob("*.parquet") if _is_race_file(f))
    if not files:
        raise FileNotFoundError(f"No race Parquet files found in {raw_dir}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        logger.debug(f"Loaded {len(df)} laps from {f.name}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total laps from {len(files)} files")
    return combined


def compute_lap_time_seconds(df: pd.DataFrame) -> pd.Series:
    """Convert LapTime to seconds, handling both timedelta and pre-converted columns."""
    if "LapTime_seconds" in df.columns:
        return df["LapTime_seconds"]
    elif "LapTime" in df.columns:
        if pd.api.types.is_timedelta64_dtype(df["LapTime"]):
            return df["LapTime"].dt.total_seconds()
        return df["LapTime"]
    raise KeyError("Neither LapTime nor LapTime_seconds found in DataFrame")


def compute_fuel_mass(
    lap_numbers: pd.Series,
    start_mass: float = 110.0,
    burn_rate: float = 1.5,
) -> pd.Series:
    """Estimate fuel mass based on lap number.

    Fuel decreases linearly from start_mass at a rate of burn_rate kg/lap.
    """
    return np.maximum(0.0, start_mass - burn_rate * (lap_numbers - 1))


def compute_stint_info(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stint number and stint lap for each driver in each race.

    A new stint starts when TyreLife resets (decreases) or Compound changes.
    """
    result = df.copy()
    result["stint_number"] = 0
    result["stint_lap"] = 0

    group_cols = []
    if "race_id" in result.columns:
        group_cols.append("race_id")
    elif "Year" in result.columns and "RoundNumber" in result.columns:
        group_cols.extend(["Year", "RoundNumber"])

    driver_col = "Driver" if "Driver" in result.columns else "driver_id"
    if driver_col in result.columns:
        group_cols.append(driver_col)

    if not group_cols:
        return result

    for _, group in result.groupby(group_cols):
        sorted_group = group.sort_values(
            "LapNumber" if "LapNumber" in group.columns else "lap_number"
        )
        sorted_idx = sorted_group.index

        stint = 1
        stint_lap = 1
        stints = [stint]
        stint_laps = [stint_lap]

        tyre_life = sorted_group["TyreLife"].values if "TyreLife" in sorted_group.columns else None

        for i in range(1, len(sorted_group)):
            if tyre_life is not None and tyre_life[i] < tyre_life[i - 1]:
                stint += 1
                stint_lap = 1
            else:
                stint_lap += 1
            stints.append(stint)
            stint_laps.append(stint_lap)

        result.loc[sorted_idx, "stint_number"] = stints
        result.loc[sorted_idx, "stint_lap"] = stint_laps

    return result


def _compute_traffic_density(
    df: pd.DataFrame,
    threshold_seconds: float = 1.5,
) -> pd.Series:
    """Count drivers within +/- threshold of each driver's lap time per race-lap.

    Returns a Series aligned with df's index.
    """
    density = pd.Series(0, index=df.index, dtype=int)

    for _, group in df.groupby(["race_id", "lap_number"]):
        times = group["lap_time_seconds"].values
        indices = group.index
        for _i, (idx, t) in enumerate(zip(indices, times, strict=False)):
            if pd.isna(t):
                continue
            count = int(np.sum(np.abs(times - t) <= threshold_seconds)) - 1  # exclude self
            density.loc[idx] = max(count, 0)

    return density


def build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Transform raw lap data into model-ready features.

    Args:
        df: Raw lap DataFrame (from FastF1 ingestion).
        config: Optional config dict. If None, loads default.

    Returns:
        DataFrame with engineered features.
    """
    if config is None:
        config = load_config()

    fuel_config = config.get("fuel", {})
    start_mass = fuel_config.get("start_mass_kg", 110.0)
    default_burn_rate = fuel_config.get("burn_rate_kg_per_lap", 1.5)
    fuel_by_circuit = config.get("fuel_by_circuit", {})

    features = pd.DataFrame()

    # Race identification
    if "Year" in df.columns and "RoundNumber" in df.columns:
        features["race_id"] = (
            df["Year"].astype(str) + "_" + df["RoundNumber"].astype(str).str.zfill(2)
        )
    # Season (year) as numeric feature — captures regulation era effects
    if "Year" in df.columns:
        features["season"] = df["Year"].astype(int)

    if "CircuitKey" in df.columns:
        features["circuit_id"] = df["CircuitKey"]

    # Circuit physical characteristics (numeric features)
    circuit_chars = config.get("circuit_characteristics", {})
    pit_loss = config.get("pit_loss", {})
    if "circuit_id" in features.columns and circuit_chars:
        circuit_keys = features["circuit_id"].map(_CIRCUIT_NAME_TO_KEY)
        features["track_length_km"] = circuit_keys.map(
            lambda k: circuit_chars.get(k, {}).get("length_km", np.nan)
        )
        features["n_corners"] = circuit_keys.map(
            lambda k: circuit_chars.get(k, {}).get("corners", np.nan)
        )
        features["tire_stress"] = circuit_keys.map(
            lambda k: circuit_chars.get(k, {}).get("tire_stress", np.nan)
        )
        features["pit_loss_seconds"] = circuit_keys.map(lambda k: pit_loss.get(k, np.nan))

    if "Driver" in df.columns:
        features["driver_id"] = df["Driver"]
    if "Team" in df.columns:
        features["constructor_id"] = df["Team"]

    # Lap info
    lap_num_col = "LapNumber" if "LapNumber" in df.columns else "lap_number"
    if lap_num_col in df.columns:
        features["lap_number"] = df[lap_num_col].astype(int)

    # Target variable
    features["lap_time_seconds"] = compute_lap_time_seconds(df)

    # Tire features
    if "TyreLife" in df.columns:
        features["tyre_life"] = df["TyreLife"].astype(float)
        features["tyre_life_sq"] = features["tyre_life"] ** 2

    # Compound
    if "Compound" in df.columns:
        features["compound"] = df["Compound"].str.upper()

    # Fuel load — use circuit-specific burn rates when available
    if "lap_number" in features.columns:
        if "circuit_id" in features.columns and fuel_by_circuit:
            circuit_keys = features["circuit_id"].map(_CIRCUIT_NAME_TO_KEY)
            burn_rates = circuit_keys.map(
                lambda k: fuel_by_circuit.get(k, default_burn_rate)
            ).fillna(default_burn_rate)
            features["fuel_mass_kg"] = np.maximum(
                0.0, start_mass - burn_rates * (features["lap_number"] - 1)
            )
        else:
            features["fuel_mass_kg"] = compute_fuel_mass(
                features["lap_number"],
                start_mass=start_mass,
                burn_rate=default_burn_rate,
            )

    # Weather features
    weather_map = {
        "AirTemp": "air_temp",
        "TrackTemp": "track_temp",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "WindSpeed": "wind_speed",
        "WindDirection": "wind_direction",
        "Rainfall": "rainfall",
    }
    for src_col, dst_col in weather_map.items():
        if src_col in df.columns:
            features[dst_col] = df[src_col]

    # Stint info (needs TyreLife and grouping columns)
    temp_for_stint = features.copy()
    if "TyreLife" in df.columns:
        temp_for_stint["TyreLife"] = df["TyreLife"]
    stint_info = compute_stint_info(temp_for_stint)
    features["stint_number"] = stint_info["stint_number"]
    features["stint_lap"] = stint_info["stint_lap"]

    # Preserve TrackStatus for downstream outlier flagging (yellow adjacency)
    if "TrackStatus" in df.columns:
        features["TrackStatus"] = df["TrackStatus"]

    # --- Traffic / Position features ---
    if "Position" in df.columns:
        features["position"] = df["Position"]

    # Gap data (from ingest gap computation)
    if "gap_ahead_seconds" in df.columns:
        features["gap_ahead_seconds"] = df["gap_ahead_seconds"]
    if "gap_behind_seconds" in df.columns:
        features["gap_behind_seconds"] = df["gap_behind_seconds"]

    # Position change per driver per race
    if (
        "position" in features.columns
        and "driver_id" in features.columns
        and "race_id" in features.columns
    ):
        features["position_change"] = features.groupby(["race_id", "driver_id"])["position"].diff()

    # Traffic density: count of drivers within +/- 1.5s on the same lap
    if (
        "lap_time_seconds" in features.columns
        and "race_id" in features.columns
        and "lap_number" in features.columns
    ):
        features["traffic_density"] = _compute_traffic_density(features, threshold_seconds=1.5)

    # --- Stint context features ---
    if "lap_number" in features.columns and "TotalLaps" in df.columns:
        features["race_progress"] = features["lap_number"] / df["TotalLaps"].values

    # Stint fraction: how far through the current stint
    if (
        "stint_lap" in features.columns
        and "race_id" in features.columns
        and "driver_id" in features.columns
    ):
        max_stint_lap = features.groupby(["race_id", "driver_id", "stint_number"])[
            "stint_lap"
        ].transform("max")
        features["stint_fraction"] = features["stint_lap"] / max_stint_lap.replace(0, 1)

    # Is final stint
    if (
        "stint_number" in features.columns
        and "race_id" in features.columns
        and "driver_id" in features.columns
    ):
        max_stint = features.groupby(["race_id", "driver_id"])["stint_number"].transform("max")
        features["is_final_stint"] = features["stint_number"] == max_stint

    # --- Track temperature delta (drying/wetting proxy) ---
    if "track_temp" in features.columns and "race_id" in features.columns:
        features["track_temp_delta"] = (
            features.groupby("race_id")["track_temp"]
            .transform(lambda x: x.rolling(3, min_periods=1).mean().diff())
            .fillna(0.0)
        )

    # --- Gap evolution features (strategy-critical for undercut/overcut) ---
    if (
        "gap_ahead_seconds" in features.columns
        and "driver_id" in features.columns
        and "race_id" in features.columns
    ):
        # Lap-over-lap gap change: positive = gap growing, negative = closing in
        features["gap_ahead_delta"] = features.groupby(["race_id", "driver_id"])[
            "gap_ahead_seconds"
        ].diff()

    if (
        "gap_behind_seconds" in features.columns
        and "driver_id" in features.columns
        and "race_id" in features.columns
    ):
        features["gap_behind_delta"] = features.groupby(["race_id", "driver_id"])[
            "gap_behind_seconds"
        ].diff()

    # --- DRS effect feature ---
    if "lap_number" in features.columns and "gap_ahead_seconds" in features.columns:
        features["drs_likely"] = (
            (features["lap_number"] >= 3) & (features["gap_ahead_seconds"] < 1.0)
        ).astype(float)

    # --- Interaction features ---
    compound_ordinal_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    if "compound" in features.columns:
        compound_ord = features["compound"].map(compound_ordinal_map)
        if "track_temp" in features.columns:
            features["compound_x_track_temp"] = compound_ord * features["track_temp"]
        if "tyre_life" in features.columns:
            features["tyre_life_x_compound"] = features["tyre_life"] * compound_ord
    if "fuel_mass_kg" in features.columns and "track_temp" in features.columns:
        features["fuel_mass_x_track_temp"] = features["fuel_mass_kg"] * features["track_temp"]
    if "humidity" in features.columns and "rainfall" in features.columns:
        features["humidity_x_rainfall"] = features["humidity"].fillna(0) * features[
            "rainfall"
        ].fillna(0).astype(float)

    # Drop rows with missing critical values
    required_cols = ["lap_time_seconds", "tyre_life", "compound"]
    present = [c for c in required_cols if c in features.columns]
    before = len(features)
    features = features.dropna(subset=present)
    # Filter invalid compound values (e.g. "NONE", "UNKNOWN")
    valid_compounds = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    if "compound" in features.columns:
        features = features[features["compound"].isin(valid_compounds)]
    if len(features) < before:
        logger.info(f"Dropped {before - len(features)} rows with null/invalid values in {present}")

    # Deduplicate: FastF1 sometimes returns duplicate lap entries
    dedup_cols = ["race_id", "driver_id", "lap_number"]
    if all(c in features.columns for c in dedup_cols):
        before_dedup = len(features)
        features = features.drop_duplicates(subset=dedup_cols, keep="first")
        if len(features) < before_dedup:
            logger.info(
                f"Removed {before_dedup - len(features)} duplicate (race, driver, lap) rows"
            )

    logger.info(f"Built features: {len(features)} laps, {len(features.columns)} columns")
    return features


def save_features(
    features: pd.DataFrame,
    output_dir: Path,
    metadata: dict | None = None,
) -> Path:
    """Save processed features to Parquet and metadata to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "laps_clean.parquet"
    features.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(features)} laps to {output_path}")

    # Save metadata
    if metadata is None:
        metadata = {
            "num_laps": len(features),
            "num_races": features["race_id"].nunique() if "race_id" in features.columns else 0,
            "columns": list(features.columns),
            "compounds": sorted(features["compound"].unique().tolist())
            if "compound" in features.columns
            else [],
            "circuits": sorted(features["circuit_id"].unique().tolist())
            if "circuit_id" in features.columns
            else [],
            "drivers": sorted(features["driver_id"].unique().tolist())
            if "driver_id" in features.columns
            else [],
        }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {meta_path}")

    return output_path
