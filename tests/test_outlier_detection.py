"""Tests for compound-aware outlier flagging and yellow adjacency."""

import numpy as np
import pandas as pd
import pytest

from f1deg.data.filters import (
    flag_outliers_compound_aware,
    flag_yellow_adjacent,
)


@pytest.fixture
def lap_data_with_outliers():
    """Lap data with some clear outliers."""
    np.random.seed(42)
    n = 100
    data = {
        "race_id": ["2024_01"] * n,
        "driver_id": np.repeat(["VER", "HAM", "LEC", "NOR"], 25),
        "lap_number": np.tile(np.arange(1, 26), 4),
        "compound": np.repeat(["SOFT", "MEDIUM", "HARD", "SOFT"], 25),
        "lap_time_seconds": np.random.normal(90, 0.5, n),
        "TrackStatus": ["1"] * n,
    }
    # Inject clear outliers
    data["lap_time_seconds"][10] = 130.0  # Way too slow
    data["lap_time_seconds"][50] = 125.0  # Way too slow
    return pd.DataFrame(data)


class TestCompoundAwareOutlierFlagging:
    def test_flags_added(self, lap_data_with_outliers):
        """Should add is_outlier and outlier_reason columns."""
        result = flag_outliers_compound_aware(lap_data_with_outliers)
        assert "is_outlier" in result.columns
        assert "outlier_reason" in result.columns

    def test_outliers_flagged(self, lap_data_with_outliers):
        """Clear outliers should be flagged."""
        result = flag_outliers_compound_aware(lap_data_with_outliers)
        assert result["is_outlier"].sum() >= 2  # At least the 2 injected outliers

    def test_normal_laps_not_flagged(self, lap_data_with_outliers):
        """Normal laps should not be flagged as outliers."""
        result = flag_outliers_compound_aware(lap_data_with_outliers)
        # Most laps should be normal
        assert result["is_outlier"].sum() < len(result) * 0.1

    def test_preserves_row_count(self, lap_data_with_outliers):
        """Should flag, not remove — same number of rows."""
        result = flag_outliers_compound_aware(lap_data_with_outliers)
        assert len(result) == len(lap_data_with_outliers)

    def test_groups_by_compound(self):
        """Different compounds should be evaluated separately."""
        data = pd.DataFrame(
            {
                "race_id": ["2024_01"] * 20,
                "compound": ["SOFT"] * 10 + ["HARD"] * 10,
                "lap_time_seconds": list(np.random.normal(88, 0.3, 10))
                + list(np.random.normal(92, 0.3, 10)),
            }
        )
        # A lap that's normal for HARD but would be outlier for SOFT
        data.loc[15, "lap_time_seconds"] = 92.8  # Normal for HARD
        result = flag_outliers_compound_aware(data)
        assert not result.loc[15, "is_outlier"]  # Should NOT be flagged


class TestYellowAdjacencyFlagging:
    def test_flags_adjacent_laps(self):
        """Laps adjacent to yellow flags should be flagged."""
        data = pd.DataFrame(
            {
                "race_id": ["2024_01"] * 10,
                "driver_id": ["VER"] * 10,
                "lap_number": list(range(1, 11)),
                "TrackStatus": ["1", "1", "1", "1", "2", "1", "1", "1", "1", "1"],
            }
        )
        result = flag_yellow_adjacent(data, adjacent_laps=1)
        # Lap 4 (before yellow) and lap 6 (after yellow) should be flagged
        assert result.loc[result["lap_number"] == 4, "is_outlier"].iloc[0]
        assert result.loc[result["lap_number"] == 6, "is_outlier"].iloc[0]

    def test_yellow_lap_not_double_flagged(self):
        """The yellow lap itself should not be flagged as adjacent."""
        data = pd.DataFrame(
            {
                "race_id": ["2024_01"] * 5,
                "driver_id": ["VER"] * 5,
                "lap_number": list(range(1, 6)),
                "TrackStatus": ["1", "1", "2", "1", "1"],
            }
        )
        result = flag_yellow_adjacent(data)
        # Lap 3 has yellow flag — should NOT be flagged by adjacency
        yellow_row = result.loc[result["lap_number"] == 3]
        assert not yellow_row["is_outlier"].iloc[0]

    def test_preserves_row_count(self):
        """Should flag, not remove."""
        data = pd.DataFrame(
            {
                "race_id": ["2024_01"] * 5,
                "driver_id": ["VER"] * 5,
                "lap_number": list(range(1, 6)),
                "TrackStatus": ["1", "2", "1", "1", "1"],
            }
        )
        result = flag_yellow_adjacent(data)
        assert len(result) == 5
