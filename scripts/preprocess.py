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

    # Convert all IP strings
    ip_cols = ["Away_SP_IP", "Home_SP_IP", "Away_Team_IP", "Home_Team_IP"]
    for col in ip_cols:
        df[f"{col}_Math"] = df[col].apply(convert_ip_to_math)

    # Calculate Bullpen Game Stats (Team Total - Starter Total)
    df["Away_Bullpen_ER"] = df["Away_Team_ER"] - df["Away_SP_ER"]
    df["Away_Bullpen_IP"] = df["Away_Team_IP_Math"] - df["Away_SP_IP_Math"]
    df["Home_Bullpen_ER"] = df["Home_Team_ER"] - df["Home_SP_ER"]
    df["Home_Bullpen_IP"] = df["Home_Team_IP_Math"] - df["Home_SP_IP_Math"]

    # ==========================================
    # 1. PROCESS TEAM ROLLING STATS (Offense + Bullpen + Rest)
    # ==========================================
    print("Calculating Team Last-30-Game OPS, Bullpen ERA, and Rest...")

    # Isolate Home and Away logs, standardizing names
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

    # REST CALCULATION: Did they play yesterday?
    team_logs["Days_Since_Last_Game"] = team_logs.groupby("Team")["Date"].diff().dt.days
    team_logs["Played_Yesterday"] = np.where(
        team_logs["Days_Since_Last_Game"] == 1, 1, 0
    )

    # ROLL RAW STATS OVER LAST 5 GAMES
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

    # CALCULATE ADVANCED METRICS FROM ROLLED SUMS
    # 1. OPS = OBP + SLG
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
    )  # Add SLG to OBP

    # 2. Strikeout Rate & Bullpen ERA
    team_logs["Team_K_Rate_L5"] = np.where(
        team_logs["Sum_AB_L5"] > 0, team_logs["Sum_K_L5"] / team_logs["Sum_AB_L5"], 0.0
    )
    team_logs["Bullpen_ERA_L5"] = np.where(
        team_logs["Sum_Bullpen_IP_L5"] > 0,
        (team_logs["Sum_Bullpen_ER_L5"] / team_logs["Sum_Bullpen_IP_L5"]) * 9,
        0.0,
    )

    # Split back and merge
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

    # ==========================================
    # 2. PROCESS PITCHER ROLLING STATS (Last 5 Starts)
    # ==========================================
    print("Calculating Pitcher PreGame ERA and K/9...")

    home_pitchers = df[
        ["Game_ID", "Date", "Home_SP", "Home_SP_ER", "Home_SP_IP_Math", "Home_SP_K"]
    ].rename(
        columns={
            "Home_SP": "Pitcher",
            "Home_SP_ER": "ER",
            "Home_SP_IP_Math": "IP",
            "Home_SP_K": "K",
        }
    )
    away_pitchers = df[
        ["Game_ID", "Date", "Away_SP", "Away_SP_ER", "Away_SP_IP_Math", "Away_SP_K"]
    ].rename(
        columns={
            "Away_SP": "Pitcher",
            "Away_SP_ER": "ER",
            "Away_SP_IP_Math": "IP",
            "Away_SP_K": "K",
        }
    )

    pitcher_logs = pd.concat([home_pitchers, away_pitchers]).sort_values(
        by=["Date", "Game_ID"]
    )

    pitcher_logs["Rolling_ER"] = pitcher_logs.groupby("Pitcher")["ER"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )
    pitcher_logs["Rolling_IP"] = pitcher_logs.groupby("Pitcher")["IP"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )
    pitcher_logs["Rolling_K"] = pitcher_logs.groupby("Pitcher")["K"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).sum()
    )

    pitcher_logs["SP_PreGame_ERA"] = np.where(
        pitcher_logs["Rolling_IP"] > 0,
        (pitcher_logs["Rolling_ER"] / pitcher_logs["Rolling_IP"]) * 9,
        0.0,
    )
    pitcher_logs["SP_PreGame_K9"] = np.where(
        pitcher_logs["Rolling_IP"] > 0,
        (pitcher_logs["Rolling_K"] / pitcher_logs["Rolling_IP"]) * 9,
        0.0,
    )

    home_p_merge = pitcher_logs[
        ["Game_ID", "Pitcher", "SP_PreGame_ERA", "SP_PreGame_K9"]
    ].rename(
        columns={
            "Pitcher": "Home_SP",
            "SP_PreGame_ERA": "Home_SP_PreGame_ERA",
            "SP_PreGame_K9": "Home_SP_PreGame_K9",
        }
    )
    away_p_merge = pitcher_logs[
        ["Game_ID", "Pitcher", "SP_PreGame_ERA", "SP_PreGame_K9"]
    ].rename(
        columns={
            "Pitcher": "Away_SP",
            "SP_PreGame_ERA": "Away_SP_PreGame_ERA",
            "SP_PreGame_K9": "Away_SP_PreGame_K9",
        }
    )

    df = df.merge(home_p_merge, on=["Game_ID", "Home_SP"], how="left")
    df = df.merge(away_p_merge, on=["Game_ID", "Away_SP"], how="left")

    # ==========================================
    # 3. CLEAN UP AND IMPUTE MISSING DATA
    # ==========================================
    print("Dropping leaky in-game stats...")

    # Keep ONLY our engineered features and identifiers
    cols_to_keep = [
        "Game_ID",
        "Date",
        "Away_Team",
        "Home_Team",
        "Away_SP",
        "Home_SP",
        "Home_Win",
        "Away_Played_Yesterday",
        "Away_Team_OPS_L5",
        "Away_Team_K_Rate_L5",
        "Away_Bullpen_ERA_L5",
        "Away_SP_PreGame_ERA",
        "Away_SP_PreGame_K9",
        "Home_Played_Yesterday",
        "Home_Team_OPS_L5",
        "Home_Team_K_Rate_L5",
        "Home_Bullpen_ERA_L5",
        "Home_SP_PreGame_ERA",
        "Home_SP_PreGame_K9",
    ]
    df = df[cols_to_keep].copy()

    print("Assigning league averages to Rookies/Debuts...")
    feature_cols = [col for col in df.columns if "L5" in col or "PreGame" in col]
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    # Played yesterday shouldn't be imputed with median, fill NaNs with 0 (no they didn't play)
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
    print("\nFinal Predictive Features:")
    print(model_data.columns.tolist())

