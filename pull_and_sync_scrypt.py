# pip install requests pandas python-dateutil numpy
import os, time, math, requests, pandas as pd, numpy as np
from typing import Dict, Any, Optional, List

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
RATE_LIMIT_PAUSE = 0.35  # be nice to Notion API

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES  = "https://fantasy.premierleague.com/api/fixtures/"
ELEMENT_SUMMARY = "https://fantasy.premierleague.com/api/element-summary/{pid}/"

# ---------- YOUR DB property names ----------
PROPERTY_MAP: Dict[str, str] = {
    "Name": "Name",
    "Player ID": "Player ID",
    "Team": "team_name",
    "Team Short": "team_short",
    "Position": "position",
    "Status": "status",
    "Price £m": "price_£m",
    "Form": "form",
    "Points/Match": "Points/Match",
    "Total Points": "total_points",
    "Selected %": "Selected %",
    "Minutes": "minutes",
    "Goals": "Goals",
    "Assists": "assists",
    "Clean Sheets": "clean_sheets",
    "Goals Conceded": "goals_conceded",
    "Pens Saved": "Pens Saved",
    "Saves": "saves",
    "Bonus": "bonus",
    "BPS": "bps",
    "Value/Form": "value_form",
    "Value/Season": "value_season",
    "Next 3 Fixtures": "next_3_fixtures",
    "Next3 FDR Avg": "next3_fdr_avg",

    # ---- Quality metrics ----
    "xG/90": "xG/90",
    "xA/90": "xA/90",
    "xGI/90": "xGI/90",
    "Points/90": "Points/90",
    "Value/90": "Value/90",
    "BPS/90": "BPS/90",
    "Bonus/90": "Bonus/90",
    "Pens Rank": "Pens Rank",
    "Corners Rank": "Corners Rank",
    "FK Rank": "FK Rank",

    # ---- Composite scores ----
    "Season Score": "Season Score",
    "Score Next 3": "Score Next 3",
    "Score Next 1": "Score Next 1",
}

# ---------- Desired Notion schema for new metrics ----------
DESIRED_NEW_PROPS: Dict[str, dict] = {
    "xG/90": {"number": {"format": "number"}},
    "xA/90": {"number": {"format": "number"}},
    "xGI/90": {"number": {"format": "number"}},
    "Points/90": {"number": {"format": "number"}},
    "Value/90": {"number": {"format": "number"}},
    "next3_fdr_avg": {"number": {"format": "number"}},
    "BPS/90": {"number": {"format": "number"}},
    "Bonus/90": {"number": {"format": "number"}},
    "Pens Rank": {"number": {"format": "number"}},
    "Corners Rank": {"number": {"format": "number"}},
    "FK Rank": {"number": {"format": "number"}},
    "Season Score": {"number": {"format": "number"}},
    "Score Next 3": {"number": {"format": "number"}},
    "Score Next 1": {"number": {"format": "number"}},
}

# ---------- Notion helpers ----------
def auth_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

def get_db_schema():
    r = requests.get(
        f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}",
        headers=auth_headers(),
        timeout=30
    )
    r.raise_for_status()
    db = r.json()
    props = db.get("properties", {})
    schema = {k: v.get("type", "rich_text") for k, v in props.items()}
    return schema

def notion_add_properties(missing_props: Dict[str, dict]):
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
    payload = {"properties": missing_props}
    r = requests.patch(url, headers=auth_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Add properties failed: {r.status_code} {r.text}")

def ensure_db_properties(schema: Dict[str, str]):
    to_add = {name: spec for name, spec in DESIRED_NEW_PROPS.items() if name not in schema}
    if to_add:
        print(f"Adding missing Notion properties: {list(to_add.keys())}")
        notion_add_properties(to_add)
        schema = get_db_schema()
    return schema

def as_title(text: str):
    return {"title": [{"type": "text", "text": {"content": text or ""}}]}

def as_rich(text: str):
    return {"rich_text": [{"type": "text", "text": {"content": text or ""}}]}

def as_number(val: Optional[float]):
    if val is None: return {"number": None}
    if isinstance(val, float) and math.isnan(val): return {"number": None}
    try:
        return {"number": float(val)}
    except:
        return {"number": None}

def as_select(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"select": None}
    return {"select": {"name": str(value)}}

def to_float(x):
    try: return float(x)
    except: return None

def extract_property_value(page_prop: dict, expected_type: str):
    if expected_type == "number":
        return page_prop.get("number", None)
    if expected_type == "title":
        arr = page_prop.get("title", [])
        return arr[0].get("plain_text") if arr else None
    if expected_type == "select":
        sel = page_prop.get("select")
        return None if sel is None else sel.get("name")
    arr = page_prop.get("rich_text", [])
    return arr[0].get("plain_text") if arr else None

# ---------- Fetch FPL data ----------
def get_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def select_existing_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Warning: bootstrap missing cols (skipped): {missing}")
    return df[existing].copy()

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

print("Fetching bootstrap data…")
bs = get_json(BOOTSTRAP)
teams_df = pd.DataFrame(bs["teams"])[["id","name","short_name"]]
types_df = pd.DataFrame(bs["element_types"])[["id","plural_name_short"]].rename(
    columns={"plural_name_short":"position"}
)
elements = pd.DataFrame(bs["elements"])

core_cols = [
    "id","web_name","first_name","second_name","team","element_type","status",
    "now_cost","form","points_per_game","total_points","selected_by_percent",
    "minutes","goals_scored","assists","clean_sheets","goals_conceded",
    "penalties_saved","saves","bonus","bps",
    "value_form","value_season",

    # Underlying + set pieces
    "expected_goals","expected_assists","expected_goal_involvements",
    "penalties_order","corners_and_indirect_freekicks_order","direct_freekicks_order",

    # Availability
    "chance_of_playing_this_round",
    "chance_of_playing_next_round",
]

players = select_existing_cols(elements, core_cols)

coerce_numeric(players, [
    "now_cost","form","points_per_game","total_points","selected_by_percent",
    "minutes","goals_scored","assists","clean_sheets","goals_conceded",
    "penalties_saved","saves","bonus","bps",
    "value_form","value_season",
    "expected_goals","expected_assists","expected_goal_involvements",
    "penalties_order","corners_and_indirect_freekicks_order","direct_freekicks_order",
    "chance_of_playing_this_round","chance_of_playing_next_round"
])

players = players.merge(teams_df, left_on="team", right_on="id", how="left", suffixes=("","_team"))
players = players.merge(types_df, left_on="element_type", right_on="id", how="left", suffixes=("","_type"))
players.drop(columns=["id_team","id_type"], inplace=True)
players.rename(columns={
    "web_name":"name",
    "name":"team_name",
    "short_name":"team_short",
    "now_cost":"price_£m_x10",
}, inplace=True)
players["price_£m"] = players["price_£m_x10"] / 10.0

print("Fetching fixtures…")
fx = pd.DataFrame(get_json(FIXTURES))
upcoming = fx[(~fx["finished"]) & (fx["event"].notna())].copy()
tid2short = dict(zip(teams_df["id"], teams_df["short_name"]))

def fdr_row(row, as_home=True):
    return f"{tid2short[row['team_h']]}(H)-{int(row['team_h_difficulty'])}" if as_home \
           else f"{tid2short[row['team_a']]}(A)-{int(row['team_a_difficulty'])}"

def next3_for_team(team_id):
    tg = upcoming[(upcoming["team_h"]==team_id)|(upcoming["team_a"]==team_id)] \
        .sort_values(["event","kickoff_time"]).head(3)
    if tg.empty:
        return "—"
    return " | ".join(fdr_row(g, as_home=(g["team_h"]==team_id)) for _, g in tg.iterrows())

def next3_fdr_avg(team_id):
    tg = upcoming[(upcoming["team_h"]==team_id)|(upcoming["team_a"]==team_id)] \
        .sort_values(["event","kickoff_time"]).head(3)
    if tg.empty:
        return None
    diffs = []
    for _, g in tg.iterrows():
        diffs.append(g["team_h_difficulty"] if g["team_h"] == team_id else g["team_a_difficulty"])
    return float(sum(diffs) / len(diffs))

def next1_fdr(team_id):
    tg = upcoming[(upcoming["team_h"]==team_id)|(upcoming["team_a"]==team_id)] \
        .sort_values(["event","kickoff_time"]).head(1)
    if tg.empty:
        return None
    g = tg.iloc[0]
    return float(g["team_h_difficulty"] if g["team_h"] == team_id else g["team_a_difficulty"])

team2next3  = {tid: next3_for_team(tid) for tid in teams_df["id"]}
team2fdravg = {tid: next3_fdr_avg(tid) for tid in teams_df["id"]}
team2fdr1   = {tid: next1_fdr(tid) for tid in teams_df["id"]}

players["next_3_fixtures"] = players["team"].map(team2next3)
players["next3_fdr_avg"]   = players["team"].map(team2fdravg)
players["next1_fdr"]       = players["team"].map(team2fdr1)

pos_map = {"Goalkeepers":"GKP","Defenders":"DEF","Midfielders":"MID","Forwards":"FWD"}
players["PositionSelect"] = players["position"].map(pos_map).fillna(players["position"])

# =============================================================================
# ---------- Derived quality metrics ----------
# =============================================================================
mins = players["minutes"].replace(0, np.nan)

players["xG_per90"]  = players["expected_goals"] * 90 / mins
players["xA_per90"]  = players["expected_assists"] * 90 / mins
players["xGI_per90"] = (players["expected_goals"] + players["expected_assists"]) * 90 / mins

players["points_per90"] = players["total_points"] * 90 / mins

# keep for display, NOT for scoring
players["value_per90"]  = players["points_per90"] / players["price_£m"]

players["bps_per90"]   = players["bps"] * 90 / mins
players["bonus_per90"] = players["bonus"] * 90 / mins

players["cs_per90"]    = players["clean_sheets"] * 90 / mins
players["saves_per90"] = players["saves"] * 90 / mins

players["on_pens_rank"]    = players["penalties_order"]
players["on_corners_rank"] = players["corners_and_indirect_freekicks_order"]
players["on_fk_rank"]      = players["direct_freekicks_order"]

# ---------- Position masks ----------
mask_gkp = players["PositionSelect"] == "GKP"
mask_def = players["PositionSelect"] == "DEF"
mask_mid = players["PositionSelect"] == "MID"
mask_fwd = players["PositionSelect"] == "FWD"

# =============================================================================
# ---------- Empirical Bayes shrinkage (stronger) ----------
# =============================================================================
PRIOR_MINS = 1000  # stronger regression for low-minutes noise

def eb_shrink_series(rate_per90: pd.Series, mins: pd.Series, prior_rate: float) -> pd.Series:
    r = pd.to_numeric(rate_per90, errors="coerce")
    m = pd.to_numeric(mins, errors="coerce").fillna(0)
    return (r * m + prior_rate * PRIOR_MINS) / (m + PRIOR_MINS)

pos_groups = players.groupby("PositionSelect")
pri_xgi90 = pos_groups["xGI_per90"].median()
pri_pts90 = pos_groups["points_per90"].median()
pri_bps90 = pos_groups["bps_per90"].median()
pri_cs90  = pos_groups["cs_per90"].median()
pri_sv90  = pos_groups["saves_per90"].median()

players["xGI_per90_shrunk"]      = players["xGI_per90"]
players["points_per90_shrunk"]   = players["points_per90"]
players["bps_per90_shrunk"]      = players["bps_per90"]
players["cs_per90_shrunk"]       = players["cs_per90"]
players["saves_per90_shrunk"]    = players["saves_per90"]

for pos, mask in [("GKP", mask_gkp), ("DEF", mask_def), ("MID", mask_mid), ("FWD", mask_fwd)]:
    mins_pos = players.loc[mask, "minutes"]
    players.loc[mask, "xGI_per90_shrunk"]    = eb_shrink_series(players.loc[mask, "xGI_per90"], mins_pos, pri_xgi90.get(pos, 0.0))
    players.loc[mask, "points_per90_shrunk"] = eb_shrink_series(players.loc[mask, "points_per90"], mins_pos, pri_pts90.get(pos, 0.0))
    players.loc[mask, "bps_per90_shrunk"]    = eb_shrink_series(players.loc[mask, "bps_per90"], mins_pos, pri_bps90.get(pos, 0.0))
    players.loc[mask, "cs_per90_shrunk"]     = eb_shrink_series(players.loc[mask, "cs_per90"], mins_pos, pri_cs90.get(pos, 0.0))
    players.loc[mask, "saves_per90_shrunk"]  = eb_shrink_series(players.loc[mask, "saves_per90"], mins_pos, pri_sv90.get(pos, 0.0))

# =============================================================================
# ---------- Long/short horizon blend ----------
# =============================================================================
RECENT_GWS = 6  # used for rate short and trend

def fetch_history(pid: int):
    try:
        return get_json(ELEMENT_SUMMARY.format(pid=pid)).get("history", [])
    except Exception:
        return []

def last_n(hist, n):
    return hist[-n:] if hist else []

print(f"Fetching element-summary history for last {RECENT_GWS} matches…")
histories = []
for pid in players["id"].astype(int).tolist():
    histories.append(fetch_history(pid))
    time.sleep(0.05)

players["recent_mins"] = [
    sum(float(m.get("minutes", 0) or 0) for m in last_n(h, RECENT_GWS)) for h in histories
]
players["recent_share"] = (players["recent_mins"] / (RECENT_GWS * 90)).clip(0, 1)

short_xgi90 = []
short_pts90 = []
short_bps90 = []

for h in histories:
    last = last_n(h, RECENT_GWS)
    mins_last = sum(float(m.get("minutes", 0) or 0) for m in last)
    if mins_last <= 0:
        short_xgi90.append(np.nan)
        short_pts90.append(np.nan)
        short_bps90.append(np.nan)
        continue
    xgi_last = sum(float(m.get("expected_goal_involvements", 0) or 0) for m in last)
    pts_last = sum(float(m.get("total_points", 0) or 0) for m in last)
    bps_last = sum(float(m.get("bps", 0) or 0) for m in last)

    short_xgi90.append(xgi_last * 90 / mins_last)
    short_pts90.append(pts_last * 90 / mins_last)
    short_bps90.append(bps_last * 90 / mins_last)

players["xGI_per90_short"]     = short_xgi90
players["points_per90_short"] = short_pts90
players["bps_per90_short"]    = short_bps90

players["xGI_per90_short_shrunk"]     = players["xGI_per90_short"]
players["points_per90_short_shrunk"] = players["points_per90_short"]
players["bps_per90_short_shrunk"]    = players["bps_per90_short"]

for pos, mask in [("GKP", mask_gkp), ("DEF", mask_def), ("MID", mask_mid), ("FWD", mask_fwd)]:
    mins_pos_recent = players.loc[mask, "recent_mins"]
    players.loc[mask, "xGI_per90_short_shrunk"] = eb_shrink_series(players.loc[mask, "xGI_per90_short"], mins_pos_recent, pri_xgi90.get(pos, 0.0))
    players.loc[mask, "points_per90_short_shrunk"] = eb_shrink_series(players.loc[mask, "points_per90_short"], mins_pos_recent, pri_pts90.get(pos, 0.0))
    players.loc[mask, "bps_per90_short_shrunk"] = eb_shrink_series(players.loc[mask, "bps_per90_short"], mins_pos_recent, pri_bps90.get(pos, 0.0))

players["xGI_per90_blend"]     = 0.70 * players["xGI_per90_shrunk"] + 0.30 * players["xGI_per90_short_shrunk"]
players["points_per90_blend"]  = 0.70 * players["points_per90_shrunk"] + 0.30 * players["points_per90_short_shrunk"]
players["bps_per90_blend"]     = 0.70 * players["bps_per90_shrunk"] + 0.30 * players["bps_per90_short_shrunk"]

players["xGI_blend_log"]    = np.log1p(players["xGI_per90_blend"])
players["pts_blend_log"]    = np.log1p(players["points_per90_blend"])
players["bps_blend_log"]    = np.log1p(players["bps_per90_blend"])
players["cs_blend_log"]     = np.log1p(players["cs_per90_shrunk"])
players["saves_blend_log"]  = np.log1p(players["saves_per90_shrunk"])

# ---------- Robust normalization ----------
def robust_minmax(s: pd.Series, lower_q=0.05, upper_q=0.975):
    s_num = pd.to_numeric(s, errors="coerce")
    lo = s_num.quantile(lower_q)
    hi = s_num.quantile(upper_q)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.0] * len(s_num), index=s_num.index)
    clipped = s_num.clip(lo, hi)
    return (clipped - lo) / (hi - lo)

norm_xgi = pd.Series(np.nan, index=players.index, dtype="float64")
norm_pts = pd.Series(np.nan, index=players.index, dtype="float64")
norm_bps = pd.Series(np.nan, index=players.index, dtype="float64")
norm_cs  = pd.Series(np.nan, index=players.index, dtype="float64")
norm_sv  = pd.Series(np.nan, index=players.index, dtype="float64")
norm_form = pd.Series(np.nan, index=players.index, dtype="float64")

for m in [mask_gkp, mask_def, mask_mid, mask_fwd]:
    norm_xgi.loc[m]  = robust_minmax(players.loc[m, "xGI_blend_log"])
    norm_pts.loc[m]  = robust_minmax(players.loc[m, "pts_blend_log"])
    norm_bps.loc[m]  = robust_minmax(players.loc[m, "bps_blend_log"])
    norm_cs.loc[m]   = robust_minmax(players.loc[m, "cs_blend_log"])
    norm_sv.loc[m]   = robust_minmax(players.loc[m, "saves_blend_log"])
    norm_form.loc[m] = robust_minmax(players.loc[m, "form"])

# =============================================================================
# ---------- Base points-potential score (no value) ----------
# =============================================================================
base_score_0_1 = pd.Series(0.0, index=players.index)

base_score_0_1.loc[mask_fwd] = (
    0.50 * norm_xgi.loc[mask_fwd] +
    0.30 * norm_pts.loc[mask_fwd] +
    0.15 * norm_bps.loc[mask_fwd] +
    0.05 * norm_form.loc[mask_fwd]
)

base_score_0_1.loc[mask_mid] = (
    0.45 * norm_xgi.loc[mask_mid] +
    0.30 * norm_pts.loc[mask_mid] +
    0.15 * norm_bps.loc[mask_mid] +
    0.10 * norm_form.loc[mask_mid]
)

base_score_0_1.loc[mask_def] = (
    0.25 * norm_xgi.loc[mask_def] +
    0.25 * norm_pts.loc[mask_def] +
    0.30 * norm_cs.loc[mask_def] +
    0.15 * norm_bps.loc[mask_def] +
    0.05 * norm_form.loc[mask_def]
)

base_score_0_1.loc[mask_gkp] = (
    0.35 * norm_pts.loc[mask_gkp] +
    0.30 * norm_sv.loc[mask_gkp] +
    0.25 * norm_cs.loc[mask_gkp] +
    0.05 * norm_bps.loc[mask_gkp] +
    0.05 * norm_form.loc[mask_gkp]
)

# =============================================================================
# ---------- Minutes multiplier (FIXED block) ----------
# =============================================================================
# Bootstrap often doesn't include games_played -> safe fallback Series
if "games_played" in players.columns:
    matches_played = pd.to_numeric(players["games_played"], errors="coerce")
    matches_played = matches_played.fillna(38).clip(lower=1)
else:
    matches_played = pd.Series(38, index=players.index, dtype="float64")

season_share = (players["minutes"] / (matches_played * 90)).clip(0, 1)

def logistic(x, k=12.0, x0=0.55):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

minutes_mult = logistic(season_share, k=12.0, x0=0.55)
minutes_mult = minutes_mult.clip(0.10, 1.00)

players["season_score_100"] = (base_score_0_1 * minutes_mult * 100).clip(0, 100)

# =============================================================================
# ---------- Rotation / chance of playing (cap downside) ----------
# =============================================================================
chance_next = pd.to_numeric(players.get("chance_of_playing_next_round"), errors="coerce")
chance_this = pd.to_numeric(players.get("chance_of_playing_this_round"), errors="coerce")
chance = chance_next.fillna(chance_this)

status = players["status"].astype(str).fillna("a")
status_fallback = pd.Series(
    np.where(status.eq("a"), 100.0, 75.0),
    index=players.index
)
chance = chance.fillna(status_fallback)
chance = (chance / 100.0).clip(0, 1)

rotation_mult = (0.70 + 0.30 * chance).clip(0.70, 1.00)

players["season_score_100"] = (players["season_score_100"] * rotation_mult).clip(0, 100)

# =============================================================================
# ---------- Ceiling boosters ----------
# =============================================================================
def rank_to_boost(rank: pd.Series, max_rank=4):
    r = pd.to_numeric(rank, errors="coerce")
    return (1.0 - (r - 1) / (max_rank - 1)).clip(0, 1).fillna(0)

pen_boost = rank_to_boost(players["on_pens_rank"], max_rank=4)
corner_boost = rank_to_boost(players["on_corners_rank"], max_rank=4)
fk_boost = rank_to_boost(players["on_fk_rank"], max_rank=4)

team_xg = players.groupby("team")["expected_goals"].transform("sum")
team_xg_z = (team_xg - team_xg.mean()) / (team_xg.std(ddof=0) + 1e-9)

ceiling_bonus = pd.Series(0.0, index=players.index)
ceiling_bonus += np.where(mask_fwd | mask_mid, 0.08 * pen_boost, 0.02 * pen_boost)
ceiling_bonus += np.where(mask_mid | mask_def, 0.05 * corner_boost + 0.04 * fk_boost, 0.01 * corner_boost)
ceiling_bonus += np.where(mask_mid | mask_fwd, 0.05 * team_xg_z.clip(-2, 2), 0.0)

trend = (players["recent_share"] - season_share).clip(0, 0.15)
ceiling_bonus += trend

ceiling_mult = (1.0 + ceiling_bonus).clip(0.90, 1.30)
players["season_score_100"] = (players["season_score_100"] * ceiling_mult).clip(0, 100)

# =============================================================================
# ---------- Fixtures (wider sensitivity) ----------
# =============================================================================
FIX_NEUTRAL = 3.0

FIXTURE_PARAMS = {
    ("MID", "next1"): {"scale": 0.15, "cap": 0.30},
    ("FWD", "next1"): {"scale": 0.15, "cap": 0.30},
    ("DEF", "next1"): {"scale": 0.10, "cap": 0.20},
    ("GKP", "next1"): {"scale": 0.10, "cap": 0.18},

    ("MID", "next3"): {"scale": 0.12, "cap": 0.25},
    ("FWD", "next3"): {"scale": 0.12, "cap": 0.25},
    ("DEF", "next3"): {"scale": 0.08, "cap": 0.18},
    ("GKP", "next3"): {"scale": 0.07, "cap": 0.15},
}

def ceiling_factor(base: pd.Series) -> pd.Series:
    b = pd.to_numeric(base, errors="coerce").fillna(0).clip(0, 1)
    return 0.60 + 0.40 * b

def fixture_mult_pos_horizon(fdr: pd.Series, base: pd.Series, pos: pd.Series, horizon: str) -> pd.Series:
    f = pd.to_numeric(fdr, errors="coerce")
    delta = (FIX_NEUTRAL - f).fillna(0)

    cf = ceiling_factor(base)
    mult = pd.Series(1.0, index=fdr.index, dtype="float64")

    for p in ["MID", "FWD", "DEF", "GKP"]:
        key = (p, horizon)
        if key not in FIXTURE_PARAMS:
            continue
        mask = pos.eq(p)
        scale = FIXTURE_PARAMS[key]["scale"]
        cap   = FIXTURE_PARAMS[key]["cap"]

        raw_bonus = delta.loc[mask] * scale
        bonus = (raw_bonus * cf.loc[mask]).clip(-cap, cap)
        mult.loc[mask] = (1.0 + bonus)

    return mult.clip(0.65, 1.35)

players["fixture_mult_next3"] = fixture_mult_pos_horizon(
    players["next3_fdr_avg"], base_score_0_1, players["PositionSelect"], "next3"
)
players["fixture_mult_next1"] = fixture_mult_pos_horizon(
    players["next1_fdr"], base_score_0_1, players["PositionSelect"], "next1"
)

players["score_next3_100"] = (players["season_score_100"] * players["fixture_mult_next3"]).clip(0, 100)
players["score_next1_100"] = (players["season_score_100"] * players["fixture_mult_next1"]).clip(0, 100)

# =============================================================================
# ---------- Recent-usage guardrails ----------
# =============================================================================
GUARD_MINUTES_LAST6 = 90
GUARD_SEASON_CAP    = 35
GUARD_SHORT_CAP     = 55

low_recent_mask = players["recent_mins"] < GUARD_MINUTES_LAST6
players.loc[low_recent_mask, "season_score_100"] = (
    players.loc[low_recent_mask, "season_score_100"].clip(upper=GUARD_SEASON_CAP)
)
players.loc[low_recent_mask, "score_next3_100"] = (
    players.loc[low_recent_mask, "score_next3_100"].clip(upper=GUARD_SHORT_CAP)
)
players.loc[low_recent_mask, "score_next1_100"] = (
    players.loc[low_recent_mask, "score_next1_100"].clip(upper=GUARD_SHORT_CAP)
)

# ---------- Final presentation cleanup ----------
for c in ["season_score_100", "score_next3_100", "score_next1_100"]:
    players[c] = pd.to_numeric(players[c], errors="coerce").fillna(0.0).clip(0, 100)

# ---------- DEBUG BLOCK ----------
DEBUG_IDS = [53, 8, 32]

def debug_player(pid: int):
    row = players.loc[players["id"] == pid]
    if row.empty:
        print(f"\nPlayer {pid} not found")
        return
    row = row.iloc[0]
    i = row.name

    pos = row["PositionSelect"]
    mins_played = float(row.get("minutes", 0) or 0)
    price = float(row.get("price_£m", 0) or 0)

    def safe_norm(s):
        v = s.loc[i]
        return float(v) if pd.notna(v) else 0.0

    n_xgi   = safe_norm(norm_xgi)
    n_pts   = safe_norm(norm_pts)
    n_bps   = safe_norm(norm_bps)
    n_form  = safe_norm(norm_form)
    n_cs    = safe_norm(norm_cs)
    n_saves = safe_norm(norm_sv)

    mmult = float(minutes_mult.loc[i]) if pd.notna(minutes_mult.loc[i]) else 1.0
    rmult = float(rotation_mult.loc[i]) if pd.notna(rotation_mult.loc[i]) else 1.0
    rmins = float(row.get("recent_mins", 0))
    rshare = float(row.get("recent_share", 0))
    fix1 = float(row.get("fixture_mult_next1", 1.0))
    fix3 = float(row.get("fixture_mult_next3", 1.0))
    cbonus = float(ceiling_bonus.loc[i]) if pd.notna(ceiling_bonus.loc[i]) else 0.0

    if pos == "MID":
        parts = {"xGI":0.45*n_xgi,"Pts":0.30*n_pts,"BPS":0.15*n_bps,"Form":0.10*n_form}
    elif pos == "FWD":
        parts = {"xGI":0.50*n_xgi,"Pts":0.30*n_pts,"BPS":0.15*n_bps,"Form":0.05*n_form}
    elif pos == "DEF":
        parts = {"xGI":0.25*n_xgi,"Pts":0.25*n_pts,"CS":0.30*n_cs,"BPS":0.15*n_bps,"Form":0.05*n_form}
    else:
        parts = {"Pts":0.35*n_pts,"Saves":0.30*n_saves,"CS":0.25*n_cs,"BPS":0.05*n_bps,"Form":0.05*n_form}

    base = float(base_score_0_1.loc[i]) if pd.notna(base_score_0_1.loc[i]) else 0.0
    season = float(players.loc[i, "season_score_100"])
    next1 = float(players.loc[i, "score_next1_100"])
    next3 = float(players.loc[i, "score_next3_100"])

    print("\n" + "="*80)
    print(f"{row['name']} (ID {pid}) | {row['team_name']} | {pos} | £{price:.1f}m | mins {mins_played:.0f}")
    print(f"status={row['status']}  chance_next={row.get('chance_of_playing_next_round', np.nan)}")
    print(f"recent_mins_last{RECENT_GWS}={rmins:.0f}  recent_share={rshare:.2f}")
    print(f"fixture_mult_next1={fix1:.3f}  fixture_mult_next3={fix3:.3f}")
    print(f"ceiling_bonus={cbonus:.3f}")
    print("- raw per90 (observed)")
    print(f"  xGI/90={row.get('xGI_per90', np.nan):.3f}  "
          f"Pts/90={row.get('points_per90', np.nan):.2f}  "
          f"BPS/90={row.get('bps_per90', np.nan):.2f}  "
          f"form={row.get('form', np.nan)}")
    print("- per90 (blend + shrunk)")
    print(f"  xGI/90_blend={row.get('xGI_per90_blend', np.nan):.3f}  "
          f"Pts/90_blend={row.get('points_per90_blend', np.nan):.2f}  "
          f"BPS/90_blend={row.get('bps_per90_blend', np.nan):.2f}")
    print("- normalized signals (0..1 within position)")
    print(f"  n_xGI={n_xgi:.3f}  n_Pts={n_pts:.3f}  n_BPS={n_bps:.3f}  n_Form={n_form:.3f}")
    if pos == "DEF":
        print(f"  n_CS={n_cs:.3f}")
    if pos == "GKP":
        print(f"  n_CS={n_cs:.3f}  n_Saves={n_saves:.3f}")
    print("- weighted base decomposition")
    for k, v in parts.items():
        print(f"  {k:>6}: {v:.4f}")
    print(f"  BASE: {base:.4f}")
    print("- multipliers")
    print(f"  minutes_mult={mmult:.3f}  rotation_mult={rmult:.3f}")
    print("- final scores")
    print(f"  SeasonScore={season:.1f}  Next3={next3:.1f}  Next1={next1:.1f}")
    print("="*80)

for pid in DEBUG_IDS:
    debug_player(pid)

# ---------- Build final df for Notion ----------
df = pd.DataFrame({
    "Player ID": players["id"].astype(int),
    "Name": players["name"].astype(str),
    "Team": players["team_name"].astype(str),
    "Team Short": players["team_short"].astype(str),
    "Position": players["PositionSelect"].astype(str),
    "Status": players["status"].astype(str),
    "Price £m": players["price_£m"].astype(float),
    "Form": players["form"].map(to_float),
    "Points/Match": players["points_per_game"].map(to_float),
    "Total Points": players["total_points"].fillna(0).astype(int),
    "Selected %": players["selected_by_percent"].map(to_float),
    "Minutes": players["minutes"].fillna(0).astype(int),
    "Goals": players["goals_scored"].fillna(0).astype(int),
    "Assists": players["assists"].fillna(0).astype(int),
    "Clean Sheets": players["clean_sheets"].fillna(0).astype(int),
    "Goals Conceded": players["goals_conceded"].fillna(0).astype(int),
    "Pens Saved": players["penalties_saved"].fillna(0).astype(int),
    "Saves": players["saves"].fillna(0).astype(int),
    "Bonus": players["bonus"].fillna(0).astype(int),
    "BPS": players["bps"].fillna(0).astype(int),
    "Value/Form": players["value_form"].map(to_float),
    "Value/Season": players["value_season"].map(to_float),
    "Next 3 Fixtures": players["next_3_fixtures"].astype(str),
    "Next3 FDR Avg": players["next3_fdr_avg"].map(to_float),

    # Quality metrics
    "xG/90": players["xG_per90"].map(to_float),
    "xA/90": players["xA_per90"].map(to_float),
    "xGI/90": players["xGI_per90"].map(to_float),
    "Points/90": players["points_per90"].map(to_float),
    "Value/90": players["value_per90"].map(to_float),
    "BPS/90": players["bps_per90"].map(to_float),
    "Bonus/90": players["bonus_per90"].map(to_float),
    "Pens Rank": players["on_pens_rank"].map(to_float),
    "Corners Rank": players["on_corners_rank"].map(to_float),
    "FK Rank": players["on_fk_rank"].map(to_float),

    # Composite scores
    "Season Score": players["season_score_100"].map(to_float),
    "Score Next 3": players["score_next3_100"].map(to_float),
    "Score Next 1": players["score_next1_100"].map(to_float),
})

# ---------- Export to CSV ----------
EXPORT_CSV_PATH = os.getenv("EXPORT_CSV_PATH", "fpl_players_scored.csv")
EXPORT_DEBUG_CSV_PATH = os.getenv("EXPORT_DEBUG_CSV_PATH", "fpl_players_debug.csv")

try:
    df.to_csv(EXPORT_CSV_PATH, index=False, encoding="utf-8")
    print(f"CSV export written to: {EXPORT_CSV_PATH}  (rows={len(df)})")
except Exception as e:
    print(f"Warn: failed to write CSV export: {e}")

debug_df = players.copy()
debug_df["n_xGI"]   = norm_xgi
debug_df["n_Pts"]   = norm_pts
debug_df["n_BPS"]   = norm_bps
debug_df["n_Form"]  = norm_form
debug_df["n_CS"]    = norm_cs
debug_df["n_Saves"] = norm_sv
debug_df["minutes_mult"] = minutes_mult
debug_df["rotation_mult"] = rotation_mult
debug_df["base_score_0_1"] = base_score_0_1
debug_df["ceiling_bonus"] = ceiling_bonus

debug_cols = [
    "id","name","team_name","team_short","PositionSelect","status","price_£m","minutes",
    "form","points_per_game","total_points",
    "next_3_fixtures","next3_fdr_avg","next1_fdr",
    "fixture_mult_next3","fixture_mult_next1",
    "chance_of_playing_this_round","chance_of_playing_next_round",
    "recent_mins","recent_share",
    "xGI_per90","points_per90","bps_per90","cs_per90","saves_per90",
    "xGI_per90_shrunk","points_per90_shrunk","bps_per90_shrunk",
    "xGI_per90_blend","points_per90_blend","bps_per90_blend",
    "n_xGI","n_Pts","n_BPS","n_Form","n_CS","n_Saves",
    "minutes_mult","rotation_mult","ceiling_bonus","base_score_0_1",
    "season_score_100","score_next3_100","score_next1_100"
]
debug_cols_existing = [c for c in debug_cols if c in debug_df.columns]
debug_df = debug_df[debug_cols_existing]

try:
    debug_df.to_csv(EXPORT_DEBUG_CSV_PATH, index=False, encoding="utf-8")
    print(f"Debug CSV export written to: {EXPORT_DEBUG_CSV_PATH}  (rows={len(debug_df)})")
except Exception as e:
    print(f"Warn: failed to write debug CSV export: {e}")

# ---------- Notion property-type aware writer ----------
def build_props(row: pd.Series, schema: Dict[str, str]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for canonical, db_name in PROPERTY_MAP.items():
        if canonical not in row:
            continue
        if db_name.lower() == "my team":
            continue
        if db_name not in schema:
            continue

        val = row[canonical]
        ptype = schema.get(db_name, "rich_text")

        if ptype == "title":
            props[db_name] = as_title(str(val) if val is not None else "")
        elif ptype == "number":
            props[db_name] = as_number(val)
        elif ptype == "select":
            props[db_name] = as_select(val)
        else:
            props[db_name] = as_rich(str(val) if val is not None else "")
    return props

def query_existing_pages_by_player_id(schema: Dict[str, str]) -> Dict[int, str]:
    pid_prop_name = PROPERTY_MAP["Player ID"]
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {"page_size": 100}
    existing: Dict[int, str] = {}
    while True:
        r = requests.post(url, headers=auth_headers(), json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        for page in data.get("results", []):
            prop = page.get("properties", {}).get(pid_prop_name, {})
            pid_raw = extract_property_value(prop, schema.get(pid_prop_name, "rich_text"))
            if pid_raw is None:
                continue
            try:
                pid_int = int(float(pid_raw)) if isinstance(pid_raw, str) else int(pid_raw)
                existing[pid_int] = page["id"]
            except Exception:
                continue
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data["next_cursor"]
        time.sleep(RATE_LIMIT_PAUSE)
    return existing

def notion_create(props: Dict[str, Any]):
    url = "https://api.notion.com/v1/pages"
    payload = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": props}
    r = requests.post(url, headers=auth_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Create failed: {r.status_code} {r.text}")

def _degrade_selects_to_rich(props: Dict[str, Any]) -> Dict[str, Any]:
    degraded: Dict[str, Any] = {}
    for k, v in props.items():
        if isinstance(v, dict) and "select" in v:
            sel = v.get("select")
            name = "" if sel is None else str(sel.get("name", ""))
            degraded[k] = as_rich(name)
        else:
            degraded[k] = v
    return degraded

def notion_update(page_id: str, props: Dict[str, Any]):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {"properties": props}
    r = requests.patch(url, headers=auth_headers(), json=payload, timeout=60)
    if r.status_code >= 400 and "select" in r.text.lower():
        degraded = _degrade_selects_to_rich(props)
        r = requests.patch(url, headers=auth_headers(), json={"properties": degraded}, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Update failed: {r.status_code} {r.text}")

def upsert_dataframe(df: pd.DataFrame):
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        raise SystemExit("Set NOTION_TOKEN and NOTION_DATABASE_ID environment vars.")

    schema = get_db_schema()
    schema = ensure_db_properties(schema)

    missing = [db_name for db_name in PROPERTY_MAP.values() if db_name not in schema.keys()]
    if missing:
        print(f"Warning: These properties are missing in your DB and will be skipped: {missing}")

    print("Indexing existing pages by Player ID…")
    existing = query_existing_pages_by_player_id(schema)
    print(f"Found {len(existing)} existing pages.")

    creates = updates = 0
    for _, row in df.iterrows():
        pid = int(row["Player ID"])
        props = build_props(row, schema)
        if pid in existing:
            notion_update(existing[pid], props); updates += 1
        else:
            notion_create(props); creates += 1
        time.sleep(RATE_LIMIT_PAUSE)

    print(f"Upsert complete. Created: {creates}, Updated: {updates}")

print(f"Prepared {len(df)} players. Upserting into Notion…")
upsert_dataframe(df)
