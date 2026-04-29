import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss


def train_baseline_model(filepath):
    print("Loading model-ready data...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # CALCULATE DIFFERENTIALS
    df["OPS_Diff"] = df["Home_Team_OPS_L5"] - df["Away_Team_OPS_L5"]
    df["K_Rate_Diff"] = df["Home_Team_K_Rate_L5"] - df["Away_Team_K_Rate_L5"]
    df["Bullpen_ERA_Diff"] = df["Away_Bullpen_ERA_L5"] - df["Home_Bullpen_ERA_L5"]
    df["SP_FIP_Diff"] = (
        df["Away_SP_PreGame_FIP"] - df["Home_SP_PreGame_FIP"]
    )  # NEW FIP DIFF
    df["SP_K9_Diff"] = df["Home_SP_PreGame_K9"] - df["Away_SP_PreGame_K9"]

    target = "Home_Win"
    features = [
        "Park_Factor",
        "OPS_Diff",
        "K_Rate_Diff",
        "Bullpen_ERA_Diff",
        "SP_FIP_Diff",  # Add FIP Diff here
        "SP_K9_Diff",
    ]

    print(f"Training on 6 Differential Features: {features}")

    train_df = df[df["Date"].dt.year < 2026].copy()
    test_df = df[df["Date"].dt.year >= 2026].copy()

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"Early stopping triggered! Model stopped at tree #{model.best_iteration}")

    # ==========================================
    # EVALUATE & GENERATE BETTING TICKET
    # ==========================================
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred_proba)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"Log Loss: {ll:.4f}")

    # ---------------------------------------------------------
    # THE BETTING TICKET: Let's see what it's actually thinking
    # ---------------------------------------------------------
    # Attach probabilities back to the test dataframe
    results_df = test_df[["Date", "Away_Team", "Home_Team", "Home_Win"]].copy()
    results_df["Home_Prob"] = y_pred_proba
    results_df["Away_Prob"] = 1 - y_pred_proba

    # Did the model correctly predict the winner?
    results_df["Correct"] = y_pred == results_df["Home_Win"]

    print("\n" + "=" * 50)
    print("🎫 TOP 5 MOST CONFIDENT HOME BETS (2026) 🎫")
    print("=" * 50)
    top_home = results_df.sort_values(by="Home_Prob", ascending=False).head(5)
    for _, row in top_home.iterrows():
        status = "✅ WON" if row["Correct"] else "❌ LOST"
        print(
            f"{row['Date'].strftime('%m-%d')}: {row['Away_Team']} @ {row['Home_Team']} | Model: {row['Home_Team']} ({row['Home_Prob']:.1%}) -> {status}"
        )

    print("\n" + "=" * 50)
    print("🎫 TOP 5 MOST CONFIDENT AWAY UPSETS (2026) 🎫")
    print("=" * 50)
    top_away = results_df.sort_values(by="Away_Prob", ascending=False).head(5)
    for _, row in top_away.iterrows():
        status = "✅ WON" if row["Correct"] else "❌ LOST"
        print(
            f"{row['Date'].strftime('%m-%d')}: {row['Away_Team']} @ {row['Home_Team']} | Model: {row['Away_Team']} ({row['Away_Prob']:.1%}) -> {status}"
        )

    print("\nFeature Importance:")
    importance = pd.DataFrame(
        {"Feature": features, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    print(importance)


if __name__ == "__main__":
    train_baseline_model("data/mlb_model_ready.csv")

