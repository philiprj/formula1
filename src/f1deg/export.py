"""Export trained models for consumption by Project B's race simulator.

Creates a self-contained model artifact with a simple prediction API.
"""

import json
from pathlib import Path

import pandas as pd

from f1deg.models.base import DegradationModel


class TireDegradationPredictor:
    """Lightweight prediction interface for exported models.

    This is the contract between Project A and Project B.
    """

    def __init__(self, model: DegradationModel, metadata: dict):
        self.model = model
        self.metadata = metadata

    def predict_lap_time(
        self,
        tyre_life: int,
        compound: str,
        fuel_mass_kg: float,
        circuit: str,
        conditions: dict | None = None,
    ) -> tuple[float, float, float]:
        """Predict a single lap time.

        Returns:
            (mean, lower_bound, upper_bound) in seconds.
        """
        if conditions is None:
            conditions = {}

        df = pd.DataFrame(
            [
                {
                    "tyre_life": float(tyre_life),
                    "tyre_life_sq": float(tyre_life**2),
                    "compound": compound.upper(),
                    "circuit_id": circuit,
                    "fuel_mass_kg": fuel_mass_kg,
                    "air_temp": conditions.get("air_temp", 25.0),
                    "track_temp": conditions.get("track_temp", 40.0),
                    "humidity": conditions.get("humidity", 50.0),
                    "wind_speed": conditions.get("wind_speed", 2.0),
                    "rainfall": conditions.get("rainfall", False),
                }
            ]
        )

        pred = self.model.predict(df)
        lower, upper = self.model.predict_interval(df)

        return float(pred[0]), float(lower[0]), float(upper[0])

    def predict_stint(
        self,
        compound: str,
        circuit: str,
        stint_length: int,
        start_fuel_kg: float = 110.0,
        burn_rate: float = 1.5,
        conditions: dict | None = None,
    ):
        """Predict lap times for an entire stint.

        Returns:
            DataFrame with lap_in_stint, predicted_lap_time, lower_bound, upper_bound.
        """
        return self.model.predict_degradation_curve(
            compound=compound,
            circuit=circuit,
            n_laps=stint_length,
            start_fuel_kg=start_fuel_kg,
            burn_rate=burn_rate,
            conditions=conditions,
        )

    def get_available_compounds(self) -> list[str]:
        result: list[str] = self.metadata.get("compounds", [])
        return result

    def get_available_circuits(self) -> list[str]:
        result: list[str] = self.metadata.get("circuits", [])
        return result


def export_model(
    model: DegradationModel,
    metadata: dict,
    output_dir: Path,
) -> Path:
    """Export a trained model as a self-contained artifact.

    Creates:
        output_dir/
            model/          — serialized model files
            metadata.json   — model metadata (compounds, circuits, features)

    Args:
        model: Trained DegradationModel.
        metadata: Dict with available compounds, circuits, etc.
        output_dir: Directory to write the artifact.

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_dir = output_dir / "model"
    model.save(model_dir)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    return output_dir


def load_predictor(
    artifact_dir: Path, model_cls: type[DegradationModel]
) -> TireDegradationPredictor:
    """Load an exported model artifact.

    Args:
        artifact_dir: Path to the exported artifact directory.
        model_cls: The model class to use for deserialization.

    Returns:
        TireDegradationPredictor instance.
    """
    model = model_cls.load(artifact_dir / "model")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return TireDegradationPredictor(model=model, metadata=metadata)
