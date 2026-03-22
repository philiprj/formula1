"""Tests for the Bayesian degradation model.

Only SVI-based fast tests are run by default.
Full MCMC tests are marked as slow and skipped in CI.
"""

import pytest

# Skip entire module if numpyro is not installed
numpyro = pytest.importorskip("numpyro")


def test_bayesian_model_instantiation():
    from f1deg.models.bayesian import BayesianDegradationModel

    model = BayesianDegradationModel()
    assert model.samples is None
    assert model.encoders == {}


@pytest.mark.slow
def test_bayesian_fit_svi(sample_processed_laps, sample_config):
    """Test SVI fitting on small synthetic data."""
    from f1deg.models.bayesian import BayesianDegradationModel

    config = sample_config.copy()
    config["model"] = {
        "svi": {"num_steps": 500, "learning_rate": 0.01},
        "priors": {
            "base_pace_mean": 90.0,
            "base_pace_sd": 10.0,
            "deg_rate_mean": 0.05,
            "deg_rate_sd": 0.03,
            "fuel_effect_mean": -0.035,
            "fuel_effect_sd": 0.01,
            "driver_offset_sd": 2.0,
            "circuit_offset_sd": 5.0,
            "obs_df": 5.0,
        },
    }

    # Use a small subset for speed
    small_df = sample_processed_laps.head(50)

    model = BayesianDegradationModel()
    model.fit_svi(small_df, config)

    # Should have learned something
    assert model.encoders is not None
    assert "compound" in model.encoders
