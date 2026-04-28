import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def tune_xgboost(filepath):
    print("Loading model-ready data...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically (Crucial for TimeSeriesSplit)
    df = df.sort_values("Date").reset_index(drop=True)

    # ==========================================
    # CALCULATE DIFFERENTIALS (The Missing Step!)
    # ==========================================
    df["OPS_Diff"] = df["Home_Team_OPS_L5"] - df["Away_Team_OPS_L5"]
    df["K_Rate_Diff"] = df["Home_Team_K_Rate_L5"] - df["Away_Team_K_Rate_L5"]
    df["Bullpen_ERA_Diff"] = (
        df["Away_Bullpen_ERA_L5"] - df["Home_Bullpen_ERA_L5"]
    )  # Flipped
    df["SP_ERA_Diff"] = df["Away_SP_PreGame_ERA"] - df["Home_SP_PreGame_ERA"]  # Flipped
    df["SP_K9_Diff"] = df["Home_SP_PreGame_K9"] - df["Away_SP_PreGame_K9"]

    features = [
        "Park_Factor",
        "OPS_Diff",
        "K_Rate_Diff",
        "Bullpen_ERA_Diff",
        "SP_ERA_Diff",
        "SP_K9_Diff",
    ]
    target = "Home_Win"

    # We only train/tune on historical data (pre-2026)
    train_df = df[df["Date"].dt.year < 2026].copy()
    X_train = train_df[features]
    y_train = train_df[target]

    print(f"Starting Grid Search on {len(X_train)} historical games...")

    # Define the "Grid" of settings we want to test
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05],
        "n_estimators": [100, 200, 300],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    # TimeSeriesSplit ensures we don't accidentally train on future data during cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Initialize the Grid Search (scoring for log loss)
    model = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=tscv,
        verbose=1,
        n_jobs=-1,
    )

    # Let it run! (This might take a minute or two)
    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 40)
    print("🏆 BEST XGBOOST SETTINGS FOUND 🏆")
    print("=" * 40)

    best_params = grid_search.best_params_
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print(f"\nBest Historical Log Loss: {-grid_search.best_score_:.4f}")
    print("\nCopy these settings into your train.py script!")


if __name__ == "__main__":
    tune_xgboost("data/mlb_model_ready.csv")
