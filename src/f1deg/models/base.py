"""Abstract base class for tire degradation models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from f1deg.config import load_config


class DegradationModel(ABC):
    """Base interface for all tire degradation models.

    All models must implement fit, predict, predict_interval, save, and load.
    The predict_degradation_curve method is the primary API for Project B's simulator.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, config: dict) -> None:
        """Train the model on processed lap data."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict lap times for given features. Returns array of predicted seconds."""
        ...

    @abstractmethod
    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict lower and upper bounds of a (1-alpha) prediction interval.

        Returns (lower_bound, upper_bound) arrays.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "DegradationModel":
        """Deserialize model from disk."""
        ...

    def predict_degradation_curve(
        self,
        compound: str,
        circuit: str,
        n_laps: int,
        start_fuel_kg: float = 110.0,
        burn_rate: float = 1.5,
        conditions: dict | None = None,
    ) -> pd.DataFrame:
        """Predict lap times for a hypothetical stint.

        This is the primary interface for Project B's RL simulator.

        Args:
            compound: Tire compound (SOFT, MEDIUM, HARD, etc.)
            circuit: Circuit identifier.
            n_laps: Number of laps in the stint.
            start_fuel_kg: Fuel mass at stint start.
            burn_rate: Fuel burn rate (kg/lap).
            conditions: Optional dict with weather conditions
                (air_temp, track_temp, humidity, etc.)

        Returns:
            DataFrame with columns:
                lap_in_stint, predicted_lap_time, lower_bound, upper_bound
        """
        if conditions is None:
            conditions = {}

        # Load circuit characteristics from config
        config = load_config()
        from f1deg.data.features import _CIRCUIT_NAME_TO_KEY

        circuit_chars = config.get("circuit_characteristics", {})
        pit_loss_map = config.get("pit_loss", {})
        circuit_key = _CIRCUIT_NAME_TO_KEY.get(circuit, "")
        chars = circuit_chars.get(circuit_key, {})

        rows = []
        track_temp = conditions.get("track_temp", 40.0)
        for lap in range(1, n_laps + 1):
            tyre_life = float(lap)
            row = {
                "tyre_life": tyre_life,
                "tyre_life_sq": tyre_life**2,
                "compound": compound.upper(),
                "circuit_id": circuit,
                "driver_id": conditions.get("driver_id", "UNKNOWN"),
                "constructor_id": conditions.get("constructor_id", "UNKNOWN"),
                "season": conditions.get("season", 2025),
                "fuel_mass_kg": max(0.0, start_fuel_kg - burn_rate * (lap - 1)),
                "air_temp": conditions.get("air_temp", 25.0),
                "track_temp": track_temp,
                "humidity": conditions.get("humidity", 50.0),
                "wind_speed": conditions.get("wind_speed", 2.0),
                "rainfall": conditions.get("rainfall", False),
                # Traffic / position — use neutral defaults for simulation
                "position": conditions.get("position", 10.0),
                "position_change": conditions.get("position_change", 0.0),
                "gap_ahead_seconds": conditions.get("gap_ahead_seconds", 2.0),
                "gap_behind_seconds": conditions.get("gap_behind_seconds", 2.0),
                "traffic_density": conditions.get("traffic_density", 0.5),
                # Stint context
                "race_progress": lap / n_laps,
                "stint_fraction": lap / n_laps,
                # Circuit characteristics
                "track_length_km": chars.get("length_km", 5.0),
                "n_corners": chars.get("corners", 15),
                "tire_stress": chars.get("tire_stress", 3),
                "pit_loss_seconds": pit_loss_map.get(circuit_key, 23.0),
                # Interaction features
                "compound_x_track_temp": 0.0,  # filled below
            }
            # compound_x_track_temp: encode compound as ordinal for interaction
            # Must match the mapping in f1deg.data.features
            compound_ord = {
                "SOFT": 0,
                "MEDIUM": 1,
                "HARD": 2,
                "INTERMEDIATE": 3,
                "WET": 4,
            }.get(compound.upper(), 1)
            row["compound_x_track_temp"] = compound_ord * track_temp
            row["tyre_life_x_compound"] = tyre_life * compound_ord
            row["fuel_mass_x_track_temp"] = row["fuel_mass_kg"] * track_temp
            row["humidity_x_rainfall"] = row["humidity"] * (1.0 if row["rainfall"] else 0.0)
            # DRS
            row["drs_likely"] = float(lap >= 3 and row["gap_ahead_seconds"] < 1.0)
            # SC/restart features (default = no recent SC)
            row["laps_since_sc_end"] = conditions.get("laps_since_sc_end", 5)
            row["laps_since_red_flag"] = conditions.get("laps_since_red_flag", 5)
            row["had_sc_this_stint"] = conditions.get("had_sc_this_stint", False)
            # Track temperature delta (default = stable conditions)
            row["track_temp_delta"] = conditions.get("track_temp_delta", 0.0)
            # Weekend calibration features (default 0 = "no practice data")
            row["circuit_baseline_pace"] = conditions.get("circuit_baseline_pace", 0.0)
            row["driver_pace_vs_field"] = conditions.get("driver_pace_vs_field", 0.0)
            row["fp_deg_rate_soft"] = conditions.get("fp_deg_rate_soft", 0.0)
            row["fp_deg_rate_medium"] = conditions.get("fp_deg_rate_medium", 0.0)
            row["fp_deg_rate_hard"] = conditions.get("fp_deg_rate_hard", 0.0)
            row["weekend_track_temp"] = conditions.get("weekend_track_temp", 0.0)
            row["quali_position"] = conditions.get("quali_position", 0.0)
            row["expected_fp3_race_delta"] = conditions.get("expected_fp3_race_delta", 3.0)
            rows.append(row)

        stint_df = pd.DataFrame(rows)
        predicted = self.predict(stint_df)
        lower, upper = self.predict_interval(stint_df)

        return pd.DataFrame(
            {
                "lap_in_stint": range(1, n_laps + 1),
                "predicted_lap_time": predicted,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
