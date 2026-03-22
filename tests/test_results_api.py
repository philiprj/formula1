"""Tests for Jolpica race results API client."""

from unittest.mock import MagicMock, patch

import pandas as pd

from f1deg.data.results import fetch_race_results, get_season_results, is_retirement


class TestIsRetirement:
    def test_finished_not_retirement(self):
        assert not is_retirement("Finished")

    def test_lapped_not_retirement(self):
        assert not is_retirement("+1 Lap")
        assert not is_retirement("+2 Laps")

    def test_engine_is_retirement(self):
        assert is_retirement("Engine")

    def test_collision_is_retirement(self):
        assert is_retirement("Collision")

    def test_accident_is_retirement(self):
        assert is_retirement("Accident")

    def test_empty_not_retirement(self):
        assert not is_retirement("")


class TestFetchRaceResults:
    @patch("f1deg.data.results.requests.get")
    def test_fetch_returns_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "MRData": {
                "total": "2",
                "RaceTable": {
                    "Races": [
                        {
                            "Results": [
                                {
                                    "Driver": {"code": "VER"},
                                    "position": "1",
                                    "grid": "1",
                                    "laps": "57",
                                    "status": "Finished",
                                },
                                {
                                    "Driver": {"code": "HAM"},
                                    "position": "R",
                                    "grid": "3",
                                    "laps": "45",
                                    "status": "Engine",
                                },
                            ]
                        }
                    ]
                },
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = fetch_race_results(2024, 1)
        assert len(results) == 2
        assert results[0]["Driver"]["code"] == "VER"

    @patch("f1deg.data.results.requests.get")
    def test_fetch_empty_race(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = fetch_race_results(2024, 99)
        assert results == []


class TestGetSeasonResults:
    @patch("f1deg.data.results.fetch_race_results")
    def test_season_results_dataframe(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "Driver": {"code": "VER"},
                "position": "1",
                "grid": "1",
                "laps": "57",
                "status": "Finished",
            }
        ]

        df = get_season_results(2024, 1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["driver_id"] == "VER"
        assert not df.iloc[0]["did_retire"]

    @patch("f1deg.data.results.fetch_race_results")
    def test_retirement_flagged(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "Driver": {"code": "HAM"},
                "position": "R",
                "grid": "3",
                "laps": "45",
                "status": "Engine",
            }
        ]

        df = get_season_results(2024, 1)
        assert df.iloc[0]["did_retire"]
        assert df.iloc[0]["laps_completed"] == 45
