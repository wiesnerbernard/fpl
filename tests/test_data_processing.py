"""Tests for data processing functions."""
import pandas as pd

from src.data_processing import (
    compute_fixture_difficulty_row,
    compute_next_n_fixtures,
    merge_team_and_position_data,
    rename_player_columns,
)


class TestMergeTeamAndPositionData:
    """Tests for merging team and position data."""

    def test_merge_success(self):
        """Test successful merge of team and position data."""
        players = pd.DataFrame(
            {
                "id": [1, 2],
                "web_name": ["Player1", "Player2"],
                "team": [1, 2],
                "element_type": [1, 2],
            }
        )
        teams = pd.DataFrame(
            {"id": [1, 2], "name": ["Arsenal", "Chelsea"], "short_name": ["ARS", "CHE"]}
        )
        types = pd.DataFrame({"id": [1, 2], "position": ["GKP", "DEF"]})

        result = merge_team_and_position_data(players, teams, types)

        assert "name" in result.columns
        assert "position" in result.columns
        assert len(result) == 2


class TestRenamePlayerColumns:
    """Tests for column renaming."""

    def test_rename_columns(self):
        """Test column renaming and price calculation."""
        players = pd.DataFrame(
            {
                "web_name": ["Salah"],
                "name": ["Liverpool"],
                "short_name": ["LIV"],
                "now_cost": [130],
            }
        )

        result = rename_player_columns(players)

        assert "name" in result.columns
        assert "team_name" in result.columns
        assert "price_£m" in result.columns
        assert result.iloc[0]["price_£m"] == 13.0


class TestFixtureDifficulty:
    """Tests for fixture difficulty calculations."""

    def test_compute_fixture_difficulty_row_home(self):
        """Test home fixture difficulty calculation."""
        row = pd.Series(
            {
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            }
        )
        team_map = {1: "ARS", 2: "CHE"}

        result = compute_fixture_difficulty_row(row, team_map, as_home=True)

        assert result == "ARS(H)-2"

    def test_compute_fixture_difficulty_row_away(self):
        """Test away fixture difficulty calculation."""
        row = pd.Series(
            {
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            }
        )
        team_map = {1: "ARS", 2: "CHE"}

        result = compute_fixture_difficulty_row(row, team_map, as_home=False)

        assert result == "CHE(A)-4"

    def test_compute_next_n_fixtures_empty(self):
        """Test next fixtures calculation with no upcoming fixtures."""
        fixtures = pd.DataFrame()
        team_map = {1: "ARS"}

        result = compute_next_n_fixtures(fixtures, 1, team_map, n=3)

        assert result == "—"

    def test_compute_next_n_fixtures_success(self):
        """Test next fixtures calculation with data."""
        fixtures = pd.DataFrame(
            {
                "team_h": [1, 2, 1],
                "team_a": [2, 1, 3],
                "team_h_difficulty": [2, 3, 2],
                "team_a_difficulty": [4, 3, 5],
                "event": [1, 2, 3],
                "kickoff_time": ["2024-01-01", "2024-01-08", "2024-01-15"],
            }
        )
        team_map = {1: "ARS", 2: "CHE", 3: "MUN"}

        result = compute_next_n_fixtures(fixtures, 1, team_map, n=2)

        assert "ARS(H)-2" in result
        assert "|" in result
