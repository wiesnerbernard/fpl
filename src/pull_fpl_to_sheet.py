"""Main script to pull FPL data and export to spreadsheet files."""
import pandas as pd

from data_processing import (
    add_fixture_data,
    merge_team_and_position_data,
    rename_player_columns,
)
from fpl_api import (
    fetch_bootstrap_data,
    fetch_fixtures_data,
    get_element_types_dataframe,
    get_elements_dataframe,
    get_teams_dataframe,
)

CORE_COLUMNS = [
    "id",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "now_cost",
    "form",
    "points_per_game",
    "total_points",
    "status",
    "selected_by_percent",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value_form",
    "value_season",
]

EXPORT_COLUMNS = [
    "id",
    "name",
    "first_name",
    "second_name",
    "team_name",
    "team_short",
    "position",
    "status",
    "price_£m",
    "form",
    "points_per_game",
    "total_points",
    "selected_by_percent",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value_form",
    "value_season",
    "next_3_fixtures",
]


def main() -> None:
    """Pull FPL data and save to Excel and CSV files."""
    print("Fetching bootstrap data…")
    bootstrap = fetch_bootstrap_data()

    # Extract core data
    teams = get_teams_dataframe(bootstrap)
    types = get_element_types_dataframe(bootstrap)
    players = get_elements_dataframe(bootstrap, CORE_COLUMNS)

    # Merge and enrich
    players = merge_team_and_position_data(players, teams, types)
    players = rename_player_columns(players)

    # Add fixtures
    print("Fetching fixtures…")
    fixtures = pd.DataFrame(fetch_fixtures_data())
    players = add_fixture_data(players, fixtures, teams)

    # Sort and select final columns
    players = players[EXPORT_COLUMNS].copy()
    players = players.sort_values(["position", "team_short", "name"]).reset_index(
        drop=True
    )

    # Save outputs
    output_xlsx = "fpl_players.xlsx"
    output_csv = "fpl_players.csv"

    players.to_excel(output_xlsx, index=False)
    players.to_csv(output_csv, index=False)

    print(f"Saved Excel: {output_xlsx} ({len(players)} rows)")
    print(f"Saved CSV:   {output_csv}  ({len(players)} rows)")


if __name__ == "__main__":
    main()
