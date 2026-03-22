"""Tests for model export and TireDegradationPredictor."""

from pathlib import Path
import tempfile

from f1deg.export import export_model, load_predictor
from f1deg.models.linear import LinearDegradationModel


def test_export_and_load(sample_processed_laps, sample_config):
    # Train a model
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    metadata = {
        "compounds": ["SOFT", "MEDIUM", "HARD"],
        "circuits": ["Bahrain", "Jeddah"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "export"
        export_model(model, metadata, output_dir)

        # Verify files exist
        assert (output_dir / "model" / "model.pkl").exists()
        assert (output_dir / "metadata.json").exists()

        # Load predictor
        predictor = load_predictor(output_dir, LinearDegradationModel)

        assert predictor.get_available_compounds() == ["SOFT", "MEDIUM", "HARD"]
        assert predictor.get_available_circuits() == ["Bahrain", "Jeddah"]


def test_predictor_predict_lap_time(sample_processed_laps, sample_config):
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    metadata = {"compounds": ["SOFT", "MEDIUM", "HARD"], "circuits": ["Bahrain"]}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "export"
        export_model(model, metadata, output_dir)
        predictor = load_predictor(output_dir, LinearDegradationModel)

        mean, lower, upper = predictor.predict_lap_time(
            tyre_life=10,
            compound="MEDIUM",
            fuel_mass_kg=80.0,
            circuit="Bahrain",
        )

        assert 60 < mean < 150
        assert lower < mean
        assert upper > mean


def test_predictor_predict_stint(sample_processed_laps, sample_config):
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    metadata = {"compounds": ["SOFT", "MEDIUM", "HARD"], "circuits": ["Bahrain"]}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "export"
        export_model(model, metadata, output_dir)
        predictor = load_predictor(output_dir, LinearDegradationModel)

        stint = predictor.predict_stint(
            compound="SOFT",
            circuit="Bahrain",
            stint_length=15,
            start_fuel_kg=100.0,
        )

        assert len(stint) == 15
        assert "predicted_lap_time" in stint.columns
        assert "lower_bound" in stint.columns
        assert "upper_bound" in stint.columns
