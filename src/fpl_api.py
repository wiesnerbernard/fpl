"""Core FPL API client and data fetching utilities."""
from typing import Any, Dict, Optional

import pandas as pd
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{pid}/"


def get_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch JSON data from a URL.

    Args:
        url: The URL to fetch from
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response as a dictionary

    Raises:
        requests.HTTPError: If the request fails
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_bootstrap_data() -> Dict[str, Any]:
    """Fetch bootstrap-static data from FPL API."""
    return get_json(BOOTSTRAP_URL)


def fetch_fixtures_data() -> Dict[str, Any]:
    """Fetch fixtures data from FPL API."""
    return get_json(FIXTURES_URL)


def fetch_element_summary(player_id: int) -> Dict[str, Any]:
    """
    Fetch detailed summary for a specific player.

    Args:
        player_id: The FPL player ID

    Returns:
        Player summary data including history
    """
    url = ELEMENT_SUMMARY_URL.format(pid=player_id)
    return get_json(url)


def get_teams_dataframe(bootstrap_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract teams data from bootstrap."""
    teams = pd.DataFrame(bootstrap_data["teams"])
    return teams[["id", "name", "short_name"]].copy()


def get_element_types_dataframe(bootstrap_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract element types (positions) from bootstrap."""
    types = pd.DataFrame(bootstrap_data["element_types"])
    return types[["id", "plural_name_short"]].rename(
        columns={"plural_name_short": "position"}
    )


def get_elements_dataframe(
    bootstrap_data: Dict[str, Any], columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Extract player elements from bootstrap data.

    Args:
        bootstrap_data: Bootstrap API response
        columns: Optional list of columns to select. If None, returns all.

    Returns:
        DataFrame of player data
    """
    elements = pd.DataFrame(bootstrap_data["elements"])
    if columns:
        existing_cols = [c for c in columns if c in elements.columns]
        return elements[existing_cols].copy()
    return elements
