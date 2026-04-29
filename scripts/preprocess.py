import pandas as pd
import numpy as np


def convert_ip_to_math(ip_str):
    if pd.isna(ip_str) or ip_str == "" or str(ip_str) == "0.0":
        return 0.0
    ip_str = str(ip_str)
    if "." in ip_str:
        full, partial = ip_str.split(".")
        return float(full) + (float(partial) / 3.0)
    return float(ip_str)


def build_predictive_dataset(filepath):
    print("Loading raw game data...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date", "Game_ID"]).reset_index(drop=True)
    # Remove duplicate game rows (can occur from double-headers fetched twice)
    before = len(df)
    df = df.drop_duplicates(subset="Game_ID", keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df)} duplicate game rows.")

    ip_cols = ["Away_SP_IP", "Home_SP_IP", "Away_Team_IP", "Home_Team_IP"]
    for col in ip_cols:
        df[f"{col}_Math"] = df[col].apply(convert_ip_to_math)

    df["Away_Bullpen_ER"] = df["Away_Team_ER"] - df["Away_SP_ER"]
    df["Away_Bullpen_IP"] = df["Away_Team_IP_Math"] - df["Away_SP_IP_Math"]
    df["Home_Bullpen_ER"] = df["Home_Team_ER"] - df["Home_SP_ER"]
    df["Home_Bullpen_IP"] = df["Home_Team_IP_Math"] - df["Home_SP_IP_Math"]

    # ==========================================
    # 1. PROCESS TEAM EWMA STATS, FATIGUE & NEW FEATURES
    # ==========================================
    print("Calculating Team EWMA Stats, Bullpen ERA, Rest, Run Diff, and Home Win Rate...")
    home_cols = [
        "Game_ID",
        "Date",
        "Home_Team",
        "Home_AB",
        "Home_Hits",
        "Home_2B",
        "Home_3B",
        "Home_HR",
        "Home_BB",
        "Home_HBP",
        "Home_SF",
        "Home_K",
        "Home_Bullpen_ER",
        "Home_Bullpen_IP",
        "Home_Score",
        "Away_Score",
        "Home_Win",
    ]
    home_teams = df[home_cols].rename(columns=lambda x: x.replace("Home_", ""))
    # After rename: "Team", "AB", "Hits", ..., "Score", and "Away_Score" stays, "Win" (was Home_Win)
    # "Away_Score" becomes the runs allowed when at home
    home_teams = home_teams.rename(columns={"Away_Score": "Runs_Allowed", "Score": "Runs_Scored"})
    home_teams["Is_Home"] = 1

    away_cols = [
        "Game_ID",
        "Date",
        "Away_Team",
        "Away_AB",
        "Away_Hits",
        "Away_2B",
        "Away_3B",
        "Away_HR",
        "Away_BB",
        "Away_HBP",
        "Away_SF",
        "Away_K",
        "Away_Bullpen_ER",
        "Away_Bullpen_IP",
        "Away_Score",
        "Home_Score",
        "Home_Win",
    ]
    away_teams = df[away_cols].rename(columns=lambda x: x.replace("Away_", ""))
    # "Score" = away runs scored, "Home_Score" = runs allowed on road, "Home_Win" = away team lost
    away_teams = away_teams.rename(columns={"Home_Score": "Runs_Allowed", "Score": "Runs_Scored"})
    # Away team won if Home_Win == 0
    away_teams["Win"] = (away_teams["Home_Win"] == 0).astype(int)
    away_teams = away_teams.drop(columns=["Home_Win"])
    away_teams["Is_Home"] = 0

    # home_teams still has "Win" column from "Home_Win" rename
    team_logs = pd.concat([home_teams, away_teams]).sort_values(by=["Date", "Game_ID"])

    # Run differential per game
    team_logs["Run_Diff"] = team_logs["Runs_Scored"] - team_logs["Runs_Allowed"]

    team_logs["Days_Since_Last_Game"] = team_logs.groupby("Team")["Date"].diff().dt.days
    # Keep full integer rest days (capped at 7 to reduce outlier noise from All-Star break, etc.)
    team_logs["Days_Rest"] = team_logs["Days_Since_Last_Game"].clip(upper=7).fillna(3)
    team_logs["Played_Yesterday"] = np.where(
        team_logs["Days_Since_Last_Game"] == 1, 1, 0
    )

    # 1A. EWMA Hitting & Bullpen Talent — two spans for recency vs. stability
    metrics = [
        "AB", "Hits", "2B", "3B", "HR", "BB", "HBP", "SF", "K",
        "Bullpen_ER", "Bullpen_IP",
    ]
    for m in metrics:
        # Long-term form (span=15, ~equivalent to original)
        team_logs[f"Sum_{m}_L15"] = team_logs.groupby("Team")[m].transform(
            lambda x: x.shift(1).ewm(span=15, min_periods=1).mean()
        )
        # Short-term form (span=5, last ~5 games)
        team_logs[f"Sum_{m}_L5"] = team_logs.groupby("Team")[m].transform(
            lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
        )

    # Rolling run differential (EWMA span=10)
    team_logs["RunDiff_EWMA"] = team_logs.groupby("Team")["Run_Diff"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )

    # Rolling win rate across ALL games (used for both home and away context)
    team_logs["Win_Rate_L20"] = team_logs.groupby("Team")["Win"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=3).mean()
    )

    # Rolling home win rate — only meaningful for home team context, so computed here
    # but stored on every row; the home team's value is what matters at inference
    home_game_logs = team_logs[team_logs["Is_Home"] == 1].copy()
    home_game_logs["Home_Win_Rate_L20"] = home_game_logs.groupby("Team")["Win"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=3).mean()
    )
    team_logs = team_logs.merge(
        home_game_logs[["Game_ID", "Team", "Home_Win_Rate_L20"]],
        on=["Game_ID", "Team"],
        how="left",
    )
    # Fill NaN for away rows with their general win rate as a fallback
    team_logs["Home_Win_Rate_L20"] = team_logs["Home_Win_Rate_L20"].fillna(team_logs["Win_Rate_L20"])

    # 1B. 3-Game Rolling Bullpen Fatigue
    team_logs["Bullpen_Fatigue_3G"] = team_logs.groupby("Team")["Bullpen_IP"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
    )

    # ---- OPS (long-term L15) ----
    obp_num = (
        team_logs["Sum_Hits_L15"] + team_logs["Sum_BB_L15"] + team_logs["Sum_HBP_L15"]
    )
    obp_den = (
        team_logs["Sum_AB_L15"]
        + team_logs["Sum_BB_L15"]
        + team_logs["Sum_HBP_L15"]
        + team_logs["Sum_SF_L15"]
    )
    team_logs["Team_OPS_L15"] = np.where(obp_den > 0, obp_num / obp_den, 0.0)
    slg_num = (
        (
            team_logs["Sum_Hits_L15"]
            - team_logs["Sum_2B_L15"]
            - team_logs["Sum_3B_L15"]
            - team_logs["Sum_HR_L15"]
        )
        + (2 * team_logs["Sum_2B_L15"])
        + (3 * team_logs["Sum_3B_L15"])
        + (4 * team_logs["Sum_HR_L15"])
    )
    team_logs["Team_OPS_L15"] += np.where(
        team_logs["Sum_AB_L15"] > 0, slg_num / team_logs["Sum_AB_L15"], 0.0
    )

    # ---- OPS (short-term L5) ----
    obp_num5 = (
        team_logs["Sum_Hits_L5"] + team_logs["Sum_BB_L5"] + team_logs["Sum_HBP_L5"]
    )
    obp_den5 = (
        team_logs["Sum_AB_L5"]
        + team_logs["Sum_BB_L5"]
        + team_logs["Sum_HBP_L5"]
        + team_logs["Sum_SF_L5"]
    )
    team_logs["Team_OPS_L5"] = np.where(obp_den5 > 0, obp_num5 / obp_den5, 0.0)
    slg_num5 = (
        (
            team_logs["Sum_Hits_L5"]
            - team_logs["Sum_2B_L5"]
            - team_logs["Sum_3B_L5"]
            - team_logs["Sum_HR_L5"]
        )
        + (2 * team_logs["Sum_2B_L5"])
        + (3 * team_logs["Sum_3B_L5"])
        + (4 * team_logs["Sum_HR_L5"])
    )
    team_logs["Team_OPS_L5"] += np.where(
        team_logs["Sum_AB_L5"] > 0, slg_num5 / team_logs["Sum_AB_L5"], 0.0
    )

    team_logs["Team_K_Rate_L15"] = np.where(
        team_logs["Sum_AB_L15"] > 0, team_logs["Sum_K_L15"] / team_logs["Sum_AB_L15"], 0.0
    )
    team_logs["Team_K_Rate_L5"] = np.where(
        team_logs["Sum_AB_L5"] > 0, team_logs["Sum_K_L5"] / team_logs["Sum_AB_L5"], 0.0
    )
    team_logs["Bullpen_ERA_L15"] = np.where(
        team_logs["Sum_Bullpen_IP_L15"] > 0,
        (team_logs["Sum_Bullpen_ER_L15"] / team_logs["Sum_Bullpen_IP_L15"]) * 9,
        0.0,
    )

    final_features = [
        "Game_ID",
        "Team",
        "Days_Rest",
        "Played_Yesterday",
        "Team_OPS_L15",
        "Team_OPS_L5",
        "Team_K_Rate_L15",
        "Team_K_Rate_L5",
        "Bullpen_ERA_L15",
        "Bullpen_Fatigue_3G",
        "RunDiff_EWMA",
        "Win_Rate_L20",
        "Home_Win_Rate_L20",
    ]
    # Only keep home-game rows for the home merge and away-game rows for the away merge
    # This prevents cartesian explosion when both teams appear in the same game
    home_merge = (
        team_logs[team_logs["Is_Home"] == 1][final_features]
        .rename(columns=lambda x: f"Home_{x}" if x not in ["Game_ID"] else x)
    )
    away_merge = (
        team_logs[team_logs["Is_Home"] == 0][final_features]
        .rename(columns=lambda x: f"Away_{x}" if x not in ["Game_ID"] else x)
    )

    df = df.merge(home_merge, on=["Game_ID", "Home_Team"], how="left")
    df = df.merge(away_merge, on=["Game_ID", "Away_Team"], how="left")

    # ==========================================
    # 2. PROCESS PITCHER FIP MATH (EWMA)
    # ==========================================
    print("Calculating Pitcher PreGame FIP and K/9...")
    home_pitchers = df[
        [
            "Game_ID",
            "Date",
            "Home_SP",
            "Home_SP_IP_Math",
            "Home_SP_K",
            "Home_SP_BB",
            "Home_SP_HR",
        ]
    ].rename(
        columns={
            "Home_SP": "Pitcher",
            "Home_SP_IP_Math": "IP",
            "Home_SP_K": "K",
            "Home_SP_BB": "BB",
            "Home_SP_HR": "HR",
        }
    )
    away_pitchers = df[
        [
            "Game_ID",
            "Date",
            "Away_SP",
            "Away_SP_IP_Math",
            "Away_SP_K",
            "Away_SP_BB",
            "Away_SP_HR",
        ]
    ].rename(
        columns={
            "Away_SP": "Pitcher",
            "Away_SP_IP_Math": "IP",
            "Away_SP_K": "K",
            "Away_SP_BB": "BB",
            "Away_SP_HR": "HR",
        }
    )

    pitcher_logs = pd.concat([home_pitchers, away_pitchers]).sort_values(
        by=["Date", "Game_ID"]
    )
    # Drop duplicate (Game_ID, Pitcher) entries from raw data duplicates
    pitcher_logs = pitcher_logs.drop_duplicates(subset=["Game_ID", "Pitcher"], keep="first")

    pitcher_logs["Rolling_IP"] = pitcher_logs.groupby("Pitcher")["IP"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    pitcher_logs["Rolling_K"] = pitcher_logs.groupby("Pitcher")["K"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    pitcher_logs["Rolling_BB"] = pitcher_logs.groupby("Pitcher")["BB"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    pitcher_logs["Rolling_HR"] = pitcher_logs.groupby("Pitcher")["HR"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )

    # Short-term pitcher form (span=4, last ~4 starts)
    pitcher_logs["Rolling_K_S"] = pitcher_logs.groupby("Pitcher")["K"].transform(
        lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
    )
    pitcher_logs["Rolling_IP_S"] = pitcher_logs.groupby("Pitcher")["IP"].transform(
        lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
    )

    fip_numerator = (
        (13 * pitcher_logs["Rolling_HR"])
        + (3 * pitcher_logs["Rolling_BB"])
        - (2 * pitcher_logs["Rolling_K"])
    )
    pitcher_logs["SP_PreGame_FIP"] = np.where(
        pitcher_logs["Rolling_IP"] > 0,
        (fip_numerator / pitcher_logs["Rolling_IP"]) + 3.20,
        0.0,
    )

    pitcher_logs["SP_PreGame_K9"] = np.where(
        pitcher_logs["Rolling_IP"] > 0,
        (pitcher_logs["Rolling_K"] / pitcher_logs["Rolling_IP"]) * 9,
        0.0,
    )

    # Short-term K/9 (hot/cold streak for pitcher)
    pitcher_logs["SP_PreGame_K9_S"] = np.where(
        pitcher_logs["Rolling_IP_S"] > 0,
        (pitcher_logs["Rolling_K_S"] / pitcher_logs["Rolling_IP_S"]) * 9,
        0.0,
    )

    # Pitcher days rest: days since last appearance
    pitcher_logs["SP_Days_Rest"] = pitcher_logs.groupby("Pitcher")["Date"].diff().dt.days.clip(upper=10).fillna(5)

    home_p_merge = pitcher_logs[
        ["Game_ID", "Pitcher", "SP_PreGame_FIP", "SP_PreGame_K9", "SP_PreGame_K9_S", "SP_Days_Rest"]
    ].rename(
        columns={
            "Pitcher": "Home_SP",
            "SP_PreGame_FIP": "Home_SP_PreGame_FIP",
            "SP_PreGame_K9": "Home_SP_PreGame_K9",
            "SP_PreGame_K9_S": "Home_SP_PreGame_K9_S",
            "SP_Days_Rest": "Home_SP_Days_Rest",
        }
    )
    away_p_merge = pitcher_logs[
        ["Game_ID", "Pitcher", "SP_PreGame_FIP", "SP_PreGame_K9", "SP_PreGame_K9_S", "SP_Days_Rest"]
    ].rename(
        columns={
            "Pitcher": "Away_SP",
            "SP_PreGame_FIP": "Away_SP_PreGame_FIP",
            "SP_PreGame_K9": "Away_SP_PreGame_K9",
            "SP_PreGame_K9_S": "Away_SP_PreGame_K9_S",
            "SP_Days_Rest": "Away_SP_Days_Rest",
        }
    )

    df = df.merge(home_p_merge, on=["Game_ID", "Home_SP"], how="left")
    df = df.merge(away_p_merge, on=["Game_ID", "Away_SP"], how="left")

    # ==========================================
    # 3. CLEAN UP AND IMPUTE MISSING DATA
    # ==========================================
    print("Adding MLB Park Factors...")
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
    df["Park_Factor"] = df["Home_Team"].map(PARK_FACTORS).fillna(100)

    print("Dropping leaky in-game stats...")

    cols_to_keep = [
        "Game_ID",
        "Date",
        "Away_Team",
        "Home_Team",
        "Away_SP",
        "Home_SP",
        "Home_Win",
        "Park_Factor",
        # Away team features — long-term
        "Away_Days_Rest",
        "Away_Team_OPS_L15",
        "Away_Team_K_Rate_L15",
        "Away_Bullpen_ERA_L15",
        "Away_Bullpen_Fatigue_3G",
        "Away_RunDiff_EWMA",
        "Away_Win_Rate_L20",
        "Away_Home_Win_Rate_L20",
        # Away team features — short-term
        "Away_Team_OPS_L5",
        "Away_Team_K_Rate_L5",
        # Away SP features
        "Away_SP_PreGame_FIP",
        "Away_SP_PreGame_K9",
        "Away_SP_PreGame_K9_S",
        "Away_SP_Days_Rest",
        # Home team features — long-term
        "Home_Days_Rest",
        "Home_Team_OPS_L15",
        "Home_Team_K_Rate_L15",
        "Home_Bullpen_ERA_L15",
        "Home_Bullpen_Fatigue_3G",
        "Home_RunDiff_EWMA",
        "Home_Win_Rate_L20",
        "Home_Home_Win_Rate_L20",
        # Home team features — short-term
        "Home_Team_OPS_L5",
        "Home_Team_K_Rate_L5",
        # Home SP features
        "Home_SP_PreGame_FIP",
        "Home_SP_PreGame_K9",
        "Home_SP_PreGame_K9_S",
        "Home_SP_Days_Rest",
    ]
    df = df[cols_to_keep].copy()

    print("Assigning league averages to Rookies/Debuts and missing values...")
    feature_cols = [col for col in df.columns if col not in
                    ["Game_ID", "Date", "Away_Team", "Home_Team", "Away_SP", "Home_SP",
                     "Home_Win", "Park_Factor"]]
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    final_df = df.dropna().reset_index(drop=True)
    print(
        f"Preprocessing complete! Dataset contains {len(final_df)} model-ready games."
    )
    return final_df


if __name__ == "__main__":
    model_data = build_predictive_dataset("data/mlb_historical_games.csv")
    model_data.to_csv("data/mlb_model_ready.csv", index=False)
