#!/usr/bin/env python3
"""
Pull current Fantasy Premier League data and write to a spreadsheet file.

Outputs:
- fpl_players.xlsx (Excel)
- fpl_players.csv  (CSV backup)

Requires:
pip install requests pandas python-dateutil openpyxl
"""

import pandas as pd
import requests

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES  = "https://fantasy.premierleague.com/api/fixtures/"

def get_json(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    print("Fetching bootstrap data…")
    bs = get_json(BOOTSTRAP)

    teams_df = pd.DataFrame(bs["teams"])[["id", "name", "short_name"]]
    types_df = pd.DataFrame(bs["element_types"])[["id", "plural_name_short"]].rename(
        columns={"plural_name_short": "position"}
    )
    elements = pd.DataFrame(bs["elements"])

    core_cols = [
        "id","web_name","first_name","second_name","team","element_type",
        "now_cost","form","points_per_game","total_points","status",
        "selected_by_percent","minutes","goals_scored","assists","clean_sheets",
        "goals_conceded","penalties_saved","penalties_missed","yellow_cards",
        "red_cards","saves","bonus","bps","influence","creativity","threat","ict_index",
        "value_form","value_season"
    ]
    players = elements[core_cols].copy()

    # Join team + position text
    players = players.merge(teams_df, left_on="team", right_on="id", how="left", suffixes=("","_team"))
    players = players.merge(types_df, left_on="element_type", right_on="id", how="left", suffixes=("","_type"))
    players.drop(columns=["id_team","id_type"], inplace=True)

    players.rename(columns={
        "web_name": "name",
        "name": "team_name",
        "short_name": "team_short",
        "now_cost": "price_£m_x10",  # FPL stores price as 10x
    }, inplace=True)
    players["price_£m"] = players["price_£m_x10"] / 10.0

    # Fixtures & next 3 fixtures per player
    print("Fetching fixtures…")
    fx = pd.DataFrame(get_json(FIXTURES))

    upcoming = fx[(~fx["finished"]) & (fx["event"].notna())].copy()
    tid2short = dict(zip(teams_df["id"], teams_df["short_name"]))

    def fdr_row(row, as_home=True):
        if as_home:
            return f"{tid2short[row['team_h']]}(H)-{int(row['team_h_difficulty'])}"
        else:
            return f"{tid2short[row['team_a']]}(A)-{int(row['team_a_difficulty'])}"

    def next3_for_team(team_id):
        team_games = upcoming[(upcoming["team_h"]==team_id)|(upcoming["team_a"]==team_id)]
        team_games = team_games.sort_values(["event","kickoff_time"]).head(3)

        labels = []
        for _, g in team_games.iterrows():
            labels.append(fdr_row(g, as_home=(g["team_h"]==team_id)))
        return " | ".join(labels)

    print("Computing next 3 fixtures per team…")
    team2next3 = {tid: next3_for_team(tid) for tid in teams_df["id"]}
    players["next_3_fixtures"] = players["team"].map(team2next3)

    # Nicely ordered export
    export_cols = [
        "id",
        "name","first_name","second_name",
        "team_name","team_short","position",
        "status","price_£m",
        "form","points_per_game","total_points","selected_by_percent",
        "minutes","goals_scored","assists","clean_sheets","goals_conceded",
        "penalties_saved","penalties_missed",
        "yellow_cards","red_cards","saves","bonus","bps",
        "influence","creativity","threat","ict_index",
        "value_form","value_season",
        "next_3_fixtures"
    ]

    players = players[export_cols].sort_values(["position","team_short","name"]).reset_index(drop=True)

    # Save
    out_xlsx = "fpl_players.xlsx"
    out_csv  = "fpl_players.csv"

    players.to_excel(out_xlsx, index=False)  # needs openpyxl
    players.to_csv(out_csv, index=False)

    print(f"Saved Excel: {out_xlsx} ({len(players)} rows)")
    print(f"Saved CSV:   {out_csv}  ({len(players)} rows)")

if __name__ == "__main__":
    main()

