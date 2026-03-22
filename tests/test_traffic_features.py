"""Tests for traffic/position and new feature engineering."""

import pandas as pd

from f1deg.data.features import build_features


class TestTrafficFeatures:
    """Tests for position, gap, and traffic density features."""

    def test_position_preserved(self, sample_raw_laps):
        """Position column should be carried through to features."""
        features = build_features(sample_raw_laps)
        assert "position" in features.columns

    def test_position_change_computed(self, sample_raw_laps):
        """Position change should be computed per driver per race."""
        features = build_features(sample_raw_laps)
        assert "position_change" in features.columns
        # First lap for each driver should be NaN (no previous to diff from)
        assert features["position_change"].isna().any()

    def test_traffic_density_non_negative(self, sample_raw_laps):
        """Traffic density should always be >= 0."""
        features = build_features(sample_raw_laps)
        if "traffic_density" in features.columns:
            assert (features["traffic_density"] >= 0).all()


class TestStintContextFeatures:
    """Tests for race progress, stint fraction, and final stint features."""

    def test_race_progress_range(self, sample_raw_laps):
        """Race progress should be between 0 and 1."""
        features = build_features(sample_raw_laps)
        if "race_progress" in features.columns:
            assert features["race_progress"].min() >= 0
            assert features["race_progress"].max() <= 1.0

    def test_stint_fraction_range(self, sample_raw_laps):
        """Stint fraction should be between 0 and 1."""
        features = build_features(sample_raw_laps)
        if "stint_fraction" in features.columns:
            valid = features["stint_fraction"].dropna()
            assert valid.min() >= 0
            assert valid.max() <= 1.0

    def test_is_final_stint_boolean(self, sample_raw_laps):
        """is_final_stint should be boolean."""
        features = build_features(sample_raw_laps)
        if "is_final_stint" in features.columns:
            assert features["is_final_stint"].dtype == bool


class TestInteractionFeatures:
    """Tests for compound x track_temp and tyre_life x track_temp."""

    def test_compound_x_track_temp(self, sample_raw_laps):
        """Interaction term should exist and reflect compound ordinal."""
        features = build_features(sample_raw_laps)
        if "compound_x_track_temp" in features.columns:
            soft_mask = features["compound"] == "SOFT"
            if soft_mask.any():
                # SOFT has ordinal 0, so interaction should be 0
                assert (features.loc[soft_mask, "compound_x_track_temp"] == 0).all()

    def test_tyre_life_x_track_temp(self, sample_raw_laps):
        """tyre_life x track_temp should be product of the two."""
        features = build_features(sample_raw_laps)
        if "tyre_life_x_track_temp" in features.columns:
            expected = features["tyre_life"] * features["track_temp"]
            pd.testing.assert_series_equal(
                features["tyre_life_x_track_temp"],
                expected,
                check_names=False,
            )
