"""
Travel & rest features for the Elo model.

Reads the raw CSV produced by collect_data.py and emits, for each game,
the rest/travel context for BOTH teams as of game time:

    - days_rest               : days since each team's previous game
    - travel_miles            : great-circle miles from each team's prev venue
    - is_day_after_night      : True if team played a night game yesterday and
                                a day game today (the classic "getaway day" trap)
    - is_doubleheader_g2      : True if this is game 2 of a DH today
    - bullpen_ip_prev         : innings the bullpen threw in the previous game
                                (team_IP - sp_IP from the prior game)
    - bullpen_ip_last3        : rolling 3-game bullpen IP

Output is one row per game with both teams' features prefixed Away_/Home_.

Penalty function at the bottom converts these into an Elo adjustment in points.
"""

import math
import pandas as pd
from datetime import datetime


EARTH_MILES = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Returns 0.0 if any coord is missing."""
    if any(v is None or pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_MILES * math.asin(math.sqrt(a))


def _ip_to_float(ip):
    """MLB stores IP as '6.2' meaning 6 and 2/3. Convert to a real float."""
    if pd.isna(ip):
        return 0.0
    s = str(ip)
    if "." in s:
        whole, frac = s.split(".")
        return int(whole) + int(frac) / 3.0
    return float(s)


def build_team_timeline(df):
    """
    Returns long-form DF with one row per (team, game) sorted by datetime.
    Each row carries the team's perspective on that game.
    """
    df = df.copy()
    df["dt"] = pd.to_datetime(df["Game_DateTime"], errors="coerce", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)

    away = pd.DataFrame(
        {
            "Game_ID": df["Game_ID"],
            "dt": df["dt"],
            "Date": df["Date"],
            "team_id": df["Away_Team_ID"],
            "team": df["Away_Team"],
            "is_home": False,
            "venue_lat": df["Venue_Lat"],
            "venue_lon": df["Venue_Lon"],
            "day_night": df["Day_Night"],
            "dh": df["Doubleheader"],
            "g_num": df["Game_Number"],
            "team_ip": df["Away_Team_IP"].map(_ip_to_float),
            "sp_ip": df["Away_SP_IP"].map(_ip_to_float),
        }
    )
    home = pd.DataFrame(
        {
            "Game_ID": df["Game_ID"],
            "dt": df["dt"],
            "Date": df["Date"],
            "team_id": df["Home_Team_ID"],
            "team": df["Home_Team"],
            "is_home": True,
            "venue_lat": df["Venue_Lat"],
            "venue_lon": df["Venue_Lon"],
            "day_night": df["Day_Night"],
            "dh": df["Doubleheader"],
            "g_num": df["Game_Number"],
            "team_ip": df["Home_Team_IP"].map(_ip_to_float),
            "sp_ip": df["Home_SP_IP"].map(_ip_to_float),
        }
    )

    timeline = pd.concat([away, home], ignore_index=True)
    timeline = timeline.sort_values(["team_id", "dt"]).reset_index(drop=True)
    timeline["bullpen_ip"] = (timeline["team_ip"] - timeline["sp_ip"]).clip(lower=0)
    return timeline


def add_rest_travel_features(timeline):
    """For each (team, game), look back at the team's previous game."""
    g = timeline.groupby("team_id", group_keys=False)

    timeline["prev_dt"] = g["dt"].shift(1)
    timeline["prev_lat"] = g["venue_lat"].shift(1)
    timeline["prev_lon"] = g["venue_lon"].shift(1)
    timeline["prev_day_night"] = g["day_night"].shift(1)
    timeline["prev_bullpen_ip"] = g["bullpen_ip"].shift(1)
    timeline["bullpen_ip_last3"] = (
        g["bullpen_ip"]
        .shift(1)
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(0, drop=True)
    )

    # Days of rest — calendar days between consecutive games
    timeline["days_rest"] = (
        (timeline["dt"] - timeline["prev_dt"]).dt.total_seconds() / 86400
    ).round(2)

    # Travel miles since previous game venue
    timeline["travel_miles"] = timeline.apply(
        lambda r: haversine_miles(
            r["prev_lat"], r["prev_lon"], r["venue_lat"], r["venue_lon"]
        ),
        axis=1,
    )

    timeline["is_day_after_night"] = (
        (timeline["prev_day_night"] == "night")
        & (timeline["day_night"] == "day")
        & (timeline["days_rest"] < 1.5)
    )
    timeline["is_doubleheader_g2"] = (timeline["dh"].isin(["Y", "S"])) & (
        timeline["g_num"] == 2
    )

    return timeline


def merge_back(df, timeline):
    """Pivot the timeline back into one row per game with Away_/Home_ prefixes."""
    keep = [
        "Game_ID",
        "team_id",
        "is_home",
        "days_rest",
        "travel_miles",
        "is_day_after_night",
        "is_doubleheader_g2",
        "prev_bullpen_ip",
        "bullpen_ip_last3",
    ]
    t = timeline[keep].copy()

    away = (
        t[~t["is_home"]]
        .drop(columns=["is_home", "team_id"])
        .add_prefix("Away_")
        .rename(columns={"Away_Game_ID": "Game_ID"})
    )
    home = (
        t[t["is_home"]]
        .drop(columns=["is_home", "team_id"])
        .add_prefix("Home_")
        .rename(columns={"Home_Game_ID": "Game_ID"})
    )

    out = df.merge(away, on="Game_ID", how="left").merge(home, on="Game_ID", how="left")
    return out


# ---------------------------------------------------------------------------
# Penalty function — Elo points to subtract from a team's game-day rating.
# These coefficients are starting points from public sabermetric work; tune
# them empirically once you backtest.
# ---------------------------------------------------------------------------
def rest_travel_penalty(
    days_rest, travel_miles, is_day_after_night, is_doubleheader_g2, bullpen_ip_last3
):
    """Return Elo points to SUBTRACT from this team's rating for this game."""
    penalty = 0.0

    # Rest: 0 days (back-to-back same calendar day, i.e. DH game 2) is worst,
    # 1 day is normal, 2+ days is a small bonus.
    if pd.isna(days_rest):
        pass
    elif days_rest < 0.5:
        penalty += 12  # DH game 2
    elif days_rest < 1.2:
        penalty += 0  # normal next-day game, baseline
    elif days_rest < 2.0:
        penalty += 0
    else:
        penalty -= 3  # extra rest, small bonus (negative penalty)

    # Travel: only really matters past ~1000 miles (cross-country trips)
    if travel_miles and travel_miles > 1000:
        penalty += min(8, (travel_miles - 1000) / 250)  # cap at 8

    # Day-after-night getaway compound effect
    if is_day_after_night:
        penalty += 6

    # DH game 2 already partially handled by days_rest; small extra
    if is_doubleheader_g2:
        penalty += 4

    # Bullpen depletion — if bullpen threw a lot in last 3 games it bites
    # tonight's win prob via late-inning collapse risk
    if not pd.isna(bullpen_ip_last3) and bullpen_ip_last3 > 12:
        penalty += min(10, (bullpen_ip_last3 - 12) * 1.5)

    return penalty


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/mlb_historical_games.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/mlb_games_with_features.csv"

    print(f"Loading {in_path}...")
    df = pd.read_csv(in_path)
    print(f"  {len(df)} games")

    print("Building team timelines...")
    timeline = build_team_timeline(df)
    timeline = add_rest_travel_features(timeline)

    print("Merging back...")
    out = merge_back(df, timeline)

    print(f"Writing {out_path}...")
    out.to_csv(out_path, index=False)
    print("Done.")
