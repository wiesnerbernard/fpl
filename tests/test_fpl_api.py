"""Tests for FPL API client functions."""
from unittest.mock import Mock, patch

from src.fpl_api import (
    get_element_types_dataframe,
    get_elements_dataframe,
    get_json,
    get_teams_dataframe,
)


class TestGetJson:
    """Tests for get_json function."""

    @patch("src.fpl_api.requests.get")
    def test_get_json_success(self, mock_get):
        """Test successful JSON fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_json("https://example.com/api")

        assert result == {"test": "data"}
        mock_get.assert_called_once_with("https://example.com/api", timeout=30)

    @patch("src.fpl_api.requests.get")
    def test_get_json_with_custom_timeout(self, mock_get):
        """Test JSON fetch with custom timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_json("https://example.com/api", timeout=60)

        mock_get.assert_called_once_with("https://example.com/api", timeout=60)


class TestDataFrameExtraction:
    """Tests for DataFrame extraction functions."""

    def test_get_teams_dataframe(self):
        """Test teams DataFrame extraction."""
        bootstrap_data = {
            "teams": [
                {"id": 1, "name": "Arsenal", "short_name": "ARS", "code": 3},
                {"id": 2, "name": "Chelsea", "short_name": "CHE", "code": 8},
            ]
        }

        result = get_teams_dataframe(bootstrap_data)

        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "short_name"]
        assert result.iloc[0]["name"] == "Arsenal"

    def test_get_element_types_dataframe(self):
        """Test element types DataFrame extraction."""
        bootstrap_data = {
            "element_types": [
                {"id": 1, "plural_name": "Goalkeepers", "plural_name_short": "GKP"},
                {"id": 2, "plural_name": "Defenders", "plural_name_short": "DEF"},
            ]
        }

        result = get_element_types_dataframe(bootstrap_data)

        assert len(result) == 2
        assert "position" in result.columns
        assert result.iloc[0]["position"] == "GKP"

    def test_get_elements_dataframe_all_columns(self):
        """Test elements DataFrame extraction without column filter."""
        bootstrap_data = {
            "elements": [
                {"id": 1, "web_name": "Salah", "team": 10, "now_cost": 130},
                {"id": 2, "web_name": "Kane", "team": 15, "now_cost": 115},
            ]
        }

        result = get_elements_dataframe(bootstrap_data)

        assert len(result) == 2
        assert "web_name" in result.columns
        assert "now_cost" in result.columns

    def test_get_elements_dataframe_with_column_filter(self):
        """Test elements DataFrame extraction with specific columns."""
        bootstrap_data = {
            "elements": [
                {"id": 1, "web_name": "Salah", "team": 10, "now_cost": 130},
            ]
        }

        result = get_elements_dataframe(bootstrap_data, columns=["id", "web_name"])

        assert len(result) == 1
        assert list(result.columns) == ["id", "web_name"]
        assert "now_cost" not in result.columns
