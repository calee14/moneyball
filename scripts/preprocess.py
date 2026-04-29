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

    ip_cols = ["Away_SP_IP", "Home_SP_IP", "Away_Team_IP", "Home_Team_IP"]
    for col in ip_cols:
        df[f"{col}_Math"] = df[col].apply(convert_ip_to_math)

    df["Away_Bullpen_ER"] = df["Away_Team_ER"] - df["Away_SP_ER"]
    df["Away_Bullpen_IP"] = df["Away_Team_IP_Math"] - df["Away_SP_IP_Math"]
    df["Home_Bullpen_ER"] = df["Home_Team_ER"] - df["Home_SP_ER"]
    df["Home_Bullpen_IP"] = df["Home_Team_IP_Math"] - df["Home_SP_IP_Math"]

    # --- TEAM STATS (Unchanged) ---
    print("Calculating Team Last-30-Game OPS, Bullpen ERA, and Rest...")
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
    ]
    home_teams = df[home_cols].rename(columns=lambda x: x.replace("Home_", ""))
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
    ]
    away_teams = df[away_cols].rename(columns=lambda x: x.replace("Away_", ""))

    team_logs = pd.concat([home_teams, away_teams]).sort_values(by=["Date", "Game_ID"])

    team_logs["Days_Since_Last_Game"] = team_logs.groupby("Team")["Date"].diff().dt.days
    team_logs["Played_Yesterday"] = np.where(
        team_logs["Days_Since_Last_Game"] == 1, 1, 0
    )

    metrics = [
        "AB",
        "Hits",
        "2B",
        "3B",
        "HR",
        "BB",
        "HBP",
        "SF",
        "K",
        "Bullpen_ER",
        "Bullpen_IP",
    ]
    for m in metrics:
        team_logs[f"Sum_{m}_L5"] = team_logs.groupby("Team")[m].transform(
            lambda x: x.shift(1).rolling(window=30, min_periods=1).sum()
        )

    obp_num = (
        team_logs["Sum_Hits_L5"] + team_logs["Sum_BB_L5"] + team_logs["Sum_HBP_L5"]
    )
    obp_den = (
        team_logs["Sum_AB_L5"]
        + team_logs["Sum_BB_L5"]
        + team_logs["Sum_HBP_L5"]
        + team_logs["Sum_SF_L5"]
    )
    team_logs["Team_OPS_L5"] = np.where(obp_den > 0, obp_num / obp_den, 0.0)

    slg_num = (
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
        team_logs["Sum_AB_L5"] > 0, slg_num / team_logs["Sum_AB_L5"], 0.0
    )

    team_logs["Team_K_Rate_L5"] = np.where(
        team_logs["Sum_AB_L5"] > 0, team_logs["Sum_K_L5"] / team_logs["Sum_AB_L5"], 0.0
    )
    team_logs["Bullpen_ERA_L5"] = np.where(
        team_logs["Sum_Bullpen_IP_L5"] > 0,
        (team_logs["Sum_Bullpen_ER_L5"] / team_logs["Sum_Bullpen_IP_L5"]) * 9,
        0.0,
    )

    final_features = [
        "Game_ID",
        "Team",
        "Played_Yesterday",
        "Team_OPS_L5",
        "Team_K_Rate_L5",
        "Bullpen_ERA_L5",
    ]
    home_merge = (
        team_logs[final_features]
        .rename(columns=lambda x: f"Home_{x}" if x not in ["Game_ID"] else x)
        .rename(columns={"Home_Team": "Home_Team"})
    )
    away_merge = (
        team_logs[final_features]
        .rename(columns=lambda x: f"Away_{x}" if x not in ["Game_ID"] else x)
        .rename(columns={"Away_Team": "Away_Team"})
    )

    df = df.merge(home_merge, on=["Game_ID", "Home_Team"], how="left")
    df = df.merge(away_merge, on=["Game_ID", "Away_Team"], how="left")

    # --- PITCHER FIP MATH ---
    print("Calculating Pitcher PreGame FIP and K/9...")

    # We now pull BB and HR into our pitcher logs
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

    # Roll the stats over the last 15 starts
    pitcher_logs["Rolling_IP"] = pitcher_logs.groupby("Pitcher")["IP"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )
    pitcher_logs["Rolling_K"] = pitcher_logs.groupby("Pitcher")["K"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )
    pitcher_logs["Rolling_BB"] = pitcher_logs.groupby("Pitcher")["BB"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )
    pitcher_logs["Rolling_HR"] = pitcher_logs.groupby("Pitcher")["HR"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )

    # FIP = ((13*HR) + (3*BB) - (2*K)) / IP + 3.20
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

    # Merge FIP back into main dataset
    home_p_merge = pitcher_logs[
        ["Game_ID", "Pitcher", "SP_PreGame_FIP", "SP_PreGame_K9"]
    ].rename(
        columns={
            "Pitcher": "Home_SP",
            "SP_PreGame_FIP": "Home_SP_PreGame_FIP",
            "SP_PreGame_K9": "Home_SP_PreGame_K9",
        }
    )
    away_p_merge = pitcher_logs[
        ["Game_ID", "Pitcher", "SP_PreGame_FIP", "SP_PreGame_K9"]
    ].rename(
        columns={
            "Pitcher": "Away_SP",
            "SP_PreGame_FIP": "Away_SP_PreGame_FIP",
            "SP_PreGame_K9": "Away_SP_PreGame_K9",
        }
    )

    df = df.merge(home_p_merge, on=["Game_ID", "Home_SP"], how="left")
    df = df.merge(away_p_merge, on=["Game_ID", "Away_SP"], how="left")

    # --- CLEANUP ---
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
        "Away_Played_Yesterday",
        "Away_Team_OPS_L5",
        "Away_Team_K_Rate_L5",
        "Away_Bullpen_ERA_L5",
        "Away_SP_PreGame_FIP",
        "Away_SP_PreGame_K9",  # Swapped ERA for FIP
        "Home_Played_Yesterday",
        "Home_Team_OPS_L5",
        "Home_Team_K_Rate_L5",
        "Home_Bullpen_ERA_L5",
        "Home_SP_PreGame_FIP",
        "Home_SP_PreGame_K9",  # Swapped ERA for FIP
    ]
    df = df[cols_to_keep].copy()

    print("Assigning league averages to Rookies/Debuts...")
    feature_cols = [col for col in df.columns if "L5" in col or "PreGame" in col]
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    df["Away_Played_Yesterday"] = df["Away_Played_Yesterday"].fillna(0)
    df["Home_Played_Yesterday"] = df["Home_Played_Yesterday"].fillna(0)

    final_df = df.dropna().reset_index(drop=True)
    print(
        f"Preprocessing complete! Dataset contains {len(final_df)} model-ready games."
    )
    return final_df


if __name__ == "__main__":
    model_data = build_predictive_dataset("data/mlb_historical_games.csv")
    model_data.to_csv("data/mlb_model_ready.csv", index=False)
