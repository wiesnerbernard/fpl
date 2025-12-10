"""Player data processing and enrichment utilities."""
from typing import Dict

import pandas as pd


def merge_team_and_position_data(
    players: pd.DataFrame, teams: pd.DataFrame, types: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge team and position information into players dataframe.

    Args:
        players: Player data
        teams: Teams lookup data
        types: Position types lookup data

    Returns:
        Enriched player DataFrame
    """
    # Merge teams
    players = players.merge(
        teams, left_on="team", right_on="id", how="left", suffixes=("", "_team")
    )

    # Merge positions
    players = players.merge(
        types, left_on="element_type", right_on="id", how="left", suffixes=("", "_type")
    )

    # Clean up duplicate ID columns
    players.drop(columns=["id_team", "id_type"], inplace=True, errors="ignore")

    return players


def rename_player_columns(players: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to more user-friendly names."""
    rename_map = {
        "web_name": "name",
        "name": "team_name",
        "short_name": "team_short",
        "now_cost": "price_£m_x10",
    }
    players.rename(columns=rename_map, inplace=True)
    players["price_£m"] = players["price_£m_x10"] / 10.0
    return players


def compute_fixture_difficulty_row(
    row: pd.Series, team_id_to_short: Dict[int, str], as_home: bool = True
) -> str:
    """
    Compute fixture difficulty string for a single fixture row.

    Args:
        row: Fixture row from fixtures DataFrame
        team_id_to_short: Mapping from team ID to short name
        as_home: Whether this is a home fixture

    Returns:
        Formatted fixture string like "ARS(H)-2"
    """
    if as_home:
        opponent = team_id_to_short[row["team_h"]]
        difficulty = int(row["team_h_difficulty"])
        return f"{opponent}(H)-{difficulty}"
    else:
        opponent = team_id_to_short[row["team_a"]]
        difficulty = int(row["team_a_difficulty"])
        return f"{opponent}(A)-{difficulty}"


def compute_next_n_fixtures(
    fixtures: pd.DataFrame,
    team_id: int,
    team_id_to_short: Dict[int, str],
    n: int = 3,
) -> str:
    """
    Compute the next N fixtures for a team.

    Args:
        fixtures: Upcoming fixtures DataFrame
        team_id: Team ID to compute fixtures for
        team_id_to_short: Mapping from team ID to short name
        n: Number of fixtures to include

    Returns:
        Formatted fixture string like "ARS(H)-2 | MUN(A)-4 | CHE(H)-3"
    """
    if fixtures.empty:
        return "—"

    team_games = fixtures[
        (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
    ].copy()

    team_games = team_games.sort_values(["event", "kickoff_time"]).head(n)

    if team_games.empty:
        return "—"

    labels = []
    for _, game in team_games.iterrows():
        is_home = game["team_h"] == team_id
        labels.append(compute_fixture_difficulty_row(game, team_id_to_short, is_home))

    return " | ".join(labels)


def add_fixture_data(
    players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """
    Add next 3 fixtures information to players.

    Args:
        players: Player DataFrame
        fixtures: Fixtures DataFrame
        teams: Teams DataFrame

    Returns:
        Players DataFrame with fixture data added
    """
    # Filter to upcoming fixtures only
    upcoming = fixtures[(~fixtures["finished"]) & (fixtures["event"].notna())].copy()

    # Create team ID to short name mapping
    team_id_to_short = dict(zip(teams["id"], teams["short_name"]))

    # Compute next 3 fixtures for each team
    team_to_fixtures = {
        tid: compute_next_n_fixtures(upcoming, tid, team_id_to_short, n=3)
        for tid in teams["id"]
    }

    # Map to players
    players["next_3_fixtures"] = players["team"].map(team_to_fixtures)

    return players
