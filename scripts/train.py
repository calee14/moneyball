import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss


def train_baseline_model(filepath):
    print("Loading model-ready data...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # ==========================================
    # FEATURE ENGINEERING: DIFFERENTIALS
    # ==========================================
    # Instead of looking at teams in isolation, we look at the GAP between them.
    # Positive numbers mean the Home Team has the advantage. Negative means Away.

    df["OPS_Diff"] = df["Home_Team_OPS_L5"] - df["Away_Team_OPS_L5"]
    df["K_Rate_Diff"] = df["Home_Team_K_Rate_L5"] - df["Away_Team_K_Rate_L5"]
    df["Bullpen_ERA_Diff"] = (
        df["Away_Bullpen_ERA_L5"] - df["Home_Bullpen_ERA_L5"]
    )  # Flipped because lower ERA is better
    df["SP_ERA_Diff"] = df["Away_SP_PreGame_ERA"] - df["Home_SP_PreGame_ERA"]  # Flipped
    df["SP_K9_Diff"] = df["Home_SP_PreGame_K9"] - df["Away_SP_PreGame_K9"]

    # Define our target
    target = "Home_Win"

    # We ONLY feed the model the differentials now.
    # We are dropping the raw team stats and the toxic 'Played_Yesterday' binary flags.
    features = [
        "OPS_Diff",
        "K_Rate_Diff",
        "Bullpen_ERA_Diff",
        "SP_ERA_Diff",
        "SP_K9_Diff",
    ]

    print(f"Training on {len(features)} Differential Features: {features}")

    # ==========================================
    # CHRONOLOGICAL SPLIT (Train on Past, Test on Future)
    # ==========================================
    train_df = df[df["Date"].dt.year < 2026].copy()
    test_df = df[df["Date"].dt.year >= 2026].copy()

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # ==========================================
    # TRAIN XGBOOST (With Regularization)
    # ==========================================
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,  # Bumped to 200
        learning_rate=0.01,  # Lowered from 0.03 to make it more cautious
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ==========================================
    # EVALUATE
    # ==========================================
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred_proba)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"Log Loss: {ll:.4f}")

    home_win_rate = y_test.mean()
    print(
        f"(Baseline: If you just guessed Home Team every game, accuracy would be {home_win_rate:.2%})"
    )

    importance = pd.DataFrame(
        {"Feature": features, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance)


if __name__ == "__main__":
    train_baseline_model("data/mlb_model_ready.csv")
