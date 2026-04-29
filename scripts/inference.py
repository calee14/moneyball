"""
inference.py — Live game win-probability predictor.

Usage:
    python scripts/inference.py \
        --away "New York Yankees" \
        --home "Boston Red Sox" \
        --away-sp "Gerrit Cole" \
        --home-sp "Tanner Houck"

How it works:
  1. Loads the preprocessed model-ready CSV (which already has all features
     computed with shift(1) so no same-game leakage).
  2. For each team/SP, grabs the LATEST available feature row — this represents
     the rolling stats entering their *next* game, which is exactly what we
     want for a future matchup.
  3. Builds a single feature vector, feeds it to the saved XGBoost model, and
     prints win probabilities.
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
DATA_PATH = "data/mlb_model_ready.csv"

PARK_FACTORS = {
    "Colorado Rockies": 112,
    "Cincinnati Reds": 107,
    "Boston Red Sox": 106,
    "Texas Rangers": 103,
    "Los Angeles Dodgers": 102,
    "Chicago White Sox": 101,
    "Atlanta Braves": 101,
    "Philadelphia Phillies": 101,
    "Los Angeles Angels": 100,
    "Houston Astros": 100,
    "Baltimore Orioles": 99,
    "Washington Nationals": 99,
    "Arizona Diamondbacks": 99,
    "Toronto Blue Jays": 99,
    "New York Yankees": 99,
    "Milwaukee Brewers": 98,
    "Chicago Cubs": 98,
    "Kansas City Royals": 98,
    "Minnesota Twins": 98,
    "Pittsburgh Pirates": 97,
    "Tampa Bay Rays": 97,
    "San Francisco Giants": 97,
    "Miami Marlins": 96,
    "New York Mets": 96,
    "St. Louis Cardinals": 96,
    "Oakland Athletics": 95,
    "Athletics": 95,
    "Detroit Tigers": 95,
    "San Diego Padres": 95,
    "Cleveland Guardians": 94,
    "Seattle Mariners": 92,
}


def load_artifacts():
    """Load the trained model and metadata."""
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"ERROR: Model not found at '{MODEL_PATH}'.\n"
            "Run 'python scripts/train.py' first to train and save the model."
        )
    if not os.path.exists(META_PATH):
        sys.exit(
            f"ERROR: Model metadata not found at '{META_PATH}'.\n"
            "Run 'python scripts/train.py' first."
        )

    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def get_latest_team_stats(df: pd.DataFrame, team: str) -> dict | None:
    """
    Return the most recent pre-game rolling stats for a team.

    The model-ready CSV stores features computed with shift(1), so the last
    row for a team already encodes stats entering *that* game. Using the
    latest row gives us the best estimate of the team's current form.
    """
    home_cols = {
        "Home_Team_OPS_L15":      "Team_OPS_L15",
        "Home_Team_OPS_L5":       "Team_OPS_L5",
        "Home_Team_K_Rate_L15":   "Team_K_Rate_L15",
        "Home_Team_K_Rate_L5":    "Team_K_Rate_L5",
        "Home_Bullpen_ERA_L15":   "Bullpen_ERA_L15",
        "Home_Bullpen_Fatigue_3G":"Bullpen_Fatigue_3G",
        "Home_RunDiff_EWMA":      "RunDiff_EWMA",
        "Home_Win_Rate_L20":      "Win_Rate_L20",
        "Home_Home_Win_Rate_L20": "Home_Win_Rate_L20",
        "Home_Days_Rest":         "Days_Rest",
    }
    away_cols = {
        "Away_Team_OPS_L15":      "Team_OPS_L15",
        "Away_Team_OPS_L5":       "Team_OPS_L5",
        "Away_Team_K_Rate_L15":   "Team_K_Rate_L15",
        "Away_Team_K_Rate_L5":    "Team_K_Rate_L5",
        "Away_Bullpen_ERA_L15":   "Bullpen_ERA_L15",
        "Away_Bullpen_Fatigue_3G":"Bullpen_Fatigue_3G",
        "Away_RunDiff_EWMA":      "RunDiff_EWMA",
        "Away_Win_Rate_L20":      "Win_Rate_L20",
        "Away_Home_Win_Rate_L20": "Home_Win_Rate_L20",
        "Away_Days_Rest":         "Days_Rest",
    }

    home_rows = df[df["Home_Team"] == team][["Date"] + list(home_cols.keys())].rename(columns=home_cols)
    away_rows = df[df["Away_Team"] == team][["Date"] + list(away_cols.keys())].rename(columns=away_cols)

    combined = pd.concat([home_rows, away_rows]).sort_values("Date")
    if combined.empty:
        return None

    latest = combined.iloc[-1]
    return {
        "Team_OPS_L15":       latest["Team_OPS_L15"],
        "Team_OPS_L5":        latest["Team_OPS_L5"],
        "Team_K_Rate_L15":    latest["Team_K_Rate_L15"],
        "Team_K_Rate_L5":     latest["Team_K_Rate_L5"],
        "Bullpen_ERA_L15":    latest["Bullpen_ERA_L15"],
        "Bullpen_Fatigue_3G": latest["Bullpen_Fatigue_3G"],
        "RunDiff_EWMA":       latest["RunDiff_EWMA"],
        "Win_Rate_L20":       latest["Win_Rate_L20"],
        "Home_Win_Rate_L20":  latest["Home_Win_Rate_L20"],
        "Days_Rest":          latest["Days_Rest"],
    }


def get_latest_sp_stats(df: pd.DataFrame, pitcher: str) -> dict | None:
    """Return the most recent pre-game rolling FIP/K9 for a starting pitcher."""
    sp_cols = {
        "Home": {
            "Home_SP_PreGame_FIP":  "SP_PreGame_FIP",
            "Home_SP_PreGame_K9":   "SP_PreGame_K9",
            "Home_SP_PreGame_K9_S": "SP_PreGame_K9_S",
            "Home_SP_Days_Rest":    "SP_Days_Rest",
        },
        "Away": {
            "Away_SP_PreGame_FIP":  "SP_PreGame_FIP",
            "Away_SP_PreGame_K9":   "SP_PreGame_K9",
            "Away_SP_PreGame_K9_S": "SP_PreGame_K9_S",
            "Away_SP_Days_Rest":    "SP_Days_Rest",
        },
    }

    home_rows = df[df["Home_SP"] == pitcher][["Date"] + list(sp_cols["Home"].keys())].rename(columns=sp_cols["Home"])
    away_rows = df[df["Away_SP"] == pitcher][["Date"] + list(sp_cols["Away"].keys())].rename(columns=sp_cols["Away"])

    combined = pd.concat([home_rows, away_rows]).sort_values("Date")
    if combined.empty:
        return None

    latest = combined.iloc[-1]
    return {
        "SP_PreGame_FIP":  latest["SP_PreGame_FIP"],
        "SP_PreGame_K9":   latest["SP_PreGame_K9"],
        "SP_PreGame_K9_S": latest["SP_PreGame_K9_S"],
        "SP_Days_Rest":    latest["SP_Days_Rest"],
    }


def _team_defaults(medians: dict) -> dict:
    """League-average fallback for a team with no historical data."""
    return {
        "Team_OPS_L15":       medians.get("OPS_Diff", 0.0),
        "Team_OPS_L5":        medians.get("OPS_Diff_S", 0.0),
        "Team_K_Rate_L15":    0.0,
        "Team_K_Rate_L5":     0.0,
        "Bullpen_ERA_L15":    medians.get("Bullpen_ERA_Diff", 4.0),
        "Bullpen_Fatigue_3G": 0.0,
        "RunDiff_EWMA":       0.0,
        "Win_Rate_L20":       0.5,
        "Home_Win_Rate_L20":  0.54,   # league-average home win rate
        "Days_Rest":          medians.get("Rest_Diff", 1.0),
    }


def _sp_defaults(medians: dict) -> dict:
    """League-average fallback for a pitcher with no historical data."""
    return {
        "SP_PreGame_FIP":  medians.get("SP_FIP_Diff", 4.0),
        "SP_PreGame_K9":   medians.get("SP_K9_Diff", 8.5),
        "SP_PreGame_K9_S": medians.get("SP_K9_Diff_S", 8.5),
        "SP_Days_Rest":    medians.get("SP_Rest_Diff", 5.0),
    }


def build_feature_vector(
    away_team: str,
    home_team: str,
    away_sp: str,
    home_sp: str,
    df: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    """Assemble the single-row feature vector for model inference."""
    medians = meta["feature_medians"]

    away_stats    = get_latest_team_stats(df, away_team)
    home_stats    = get_latest_team_stats(df, home_team)
    away_sp_stats = get_latest_sp_stats(df, away_sp)
    home_sp_stats = get_latest_sp_stats(df, home_sp)

    if away_stats is None:
        print(f"WARNING: No historical data for away team '{away_team}'. Using league medians.")
        away_stats = _team_defaults(medians)
    if home_stats is None:
        print(f"WARNING: No historical data for home team '{home_team}'. Using league medians.")
        home_stats = _team_defaults(medians)
    if away_sp_stats is None:
        print(f"WARNING: No historical data for away SP '{away_sp}'. Using league medians.")
        away_sp_stats = _sp_defaults(medians)
    if home_sp_stats is None:
        print(f"WARNING: No historical data for home SP '{home_sp}'. Using league medians.")
        home_sp_stats = _sp_defaults(medians)

    park_factor = PARK_FACTORS.get(home_team, 100)

    row = {
        # Park context
        "Park_Factor":          park_factor,
        # Long-term differentials
        "OPS_Diff":             home_stats["Team_OPS_L15"]    - away_stats["Team_OPS_L15"],
        "K_Rate_Diff":          home_stats["Team_K_Rate_L15"] - away_stats["Team_K_Rate_L15"],
        "Bullpen_ERA_Diff":     away_stats["Bullpen_ERA_L15"] - home_stats["Bullpen_ERA_L15"],
        "SP_FIP_Diff":          away_sp_stats["SP_PreGame_FIP"]  - home_sp_stats["SP_PreGame_FIP"],
        "SP_K9_Diff":           home_sp_stats["SP_PreGame_K9"]   - away_sp_stats["SP_PreGame_K9"],
        "Bullpen_Fatigue_Diff": away_stats["Bullpen_Fatigue_3G"] - home_stats["Bullpen_Fatigue_3G"],
        # Short-term differentials
        "OPS_Diff_S":           home_stats["Team_OPS_L5"]      - away_stats["Team_OPS_L5"],
        "K_Rate_Diff_S":        home_stats["Team_K_Rate_L5"]   - away_stats["Team_K_Rate_L5"],
        "SP_K9_Diff_S":         home_sp_stats["SP_PreGame_K9_S"] - away_sp_stats["SP_PreGame_K9_S"],
        # Run differential
        "RunDiff_Diff":         home_stats["RunDiff_EWMA"] - away_stats["RunDiff_EWMA"],
        # Rest
        "Rest_Diff":            home_stats["Days_Rest"]    - away_stats["Days_Rest"],
        "SP_Rest_Diff":         home_sp_stats["SP_Days_Rest"] - away_sp_stats["SP_Days_Rest"],
        # Home field advantage
        "Home_Win_Rate":        home_stats["Home_Win_Rate_L20"],
    }

    features = meta["features"]
    # Fill any missing keys with training medians
    for feat in features:
        if feat not in row or pd.isna(row[feat]):
            row[feat] = medians.get(feat, 0.0)

    return pd.DataFrame([row])[features]


def predict_game(
    away_team: str,
    home_team: str,
    away_sp: str,
    home_sp: str,
):
    model, meta = load_artifacts()

    print(f"\nLoading preprocessed data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"\nBuilding feature vector for: {away_team} @ {home_team}")
    print(f"  Away SP: {away_sp}")
    print(f"  Home SP: {home_sp}")

    X = build_feature_vector(away_team, home_team, away_sp, home_sp, df, meta)

    print("\nFeature vector:")
    print(X.to_string(index=False))

    home_prob = model.predict_proba(X)[0][1]
    away_prob = 1.0 - home_prob

    print("\n" + "=" * 50)
    print(f"  MATCHUP: {away_team} @ {home_team}")
    print("=" * 50)
    print(f"  {home_team} (Home) WIN probability : {home_prob:.1%}")
    print(f"  {away_team} (Away) WIN probability : {away_prob:.1%}")

    if home_prob > away_prob:
        edge = home_prob - 0.5
        print(f"\n  Model favors: {home_team} (home) by {edge:.1%} edge")
    else:
        edge = away_prob - 0.5
        print(f"\n  Model favors: {away_team} (away) by {edge:.1%} edge")

    print("=" * 50)
    return home_prob, away_prob


def main():
    parser = argparse.ArgumentParser(
        description="Predict win probability for an upcoming MLB game."
    )
    parser.add_argument("--away", required=True, help='Away team name, e.g. "New York Yankees"')
    parser.add_argument("--home", required=True, help='Home team name, e.g. "Boston Red Sox"')
    parser.add_argument("--away-sp", required=True, dest="away_sp", help="Away starting pitcher full name")
    parser.add_argument("--home-sp", required=True, dest="home_sp", help="Home starting pitcher full name")
    args = parser.parse_args()

    predict_game(
        away_team=args.away,
        home_team=args.home,
        away_sp=args.away_sp,
        home_sp=args.home_sp,
    )


if __name__ == "__main__":
    main()
