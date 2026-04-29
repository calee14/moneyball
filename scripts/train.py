import os
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def build_features(df):
    """Compute all differential and composite features from the model-ready dataframe."""

    # ---- Long-term differentials ----
    df["OPS_Diff"] = df["Home_Team_OPS_L15"] - df["Away_Team_OPS_L15"]
    df["K_Rate_Diff"] = df["Home_Team_K_Rate_L15"] - df["Away_Team_K_Rate_L15"]
    df["Bullpen_ERA_Diff"] = df["Away_Bullpen_ERA_L15"] - df["Home_Bullpen_ERA_L15"]
    df["SP_FIP_Diff"] = df["Away_SP_PreGame_FIP"] - df["Home_SP_PreGame_FIP"]
    df["SP_K9_Diff"] = df["Home_SP_PreGame_K9"] - df["Away_SP_PreGame_K9"]
    df["Bullpen_Fatigue_Diff"] = df["Away_Bullpen_Fatigue_3G"] - df["Home_Bullpen_Fatigue_3G"]

    # ---- Short-term (hot/cold streak) differentials ----
    df["OPS_Diff_S"] = df["Home_Team_OPS_L5"] - df["Away_Team_OPS_L5"]
    df["K_Rate_Diff_S"] = df["Home_Team_K_Rate_L5"] - df["Away_Team_K_Rate_L5"]
    df["SP_K9_Diff_S"] = df["Home_SP_PreGame_K9_S"] - df["Away_SP_PreGame_K9_S"]

    # ---- Run differential (composite team quality signal) ----
    df["RunDiff_Diff"] = df["Home_RunDiff_EWMA"] - df["Away_RunDiff_EWMA"]

    # ---- Rest advantage ----
    df["Rest_Diff"] = df["Home_Days_Rest"] - df["Away_Days_Rest"]
    df["SP_Rest_Diff"] = df["Home_SP_Days_Rest"] - df["Away_SP_Days_Rest"]

    # ---- Home team's established home win rate ----
    # Use the home team's rolling home win rate directly (already home-specific)
    df["Home_Win_Rate"] = df["Home_Home_Win_Rate_L20"]

    return df


def train_baseline_model(filepath):
    print("Loading model-ready data...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = build_features(df)

    target = "Home_Win"
    features = [
        # Park context
        "Park_Factor",
        # Long-term form differentials
        "OPS_Diff",
        "K_Rate_Diff",
        "Bullpen_ERA_Diff",
        "SP_FIP_Diff",
        "SP_K9_Diff",
        "Bullpen_Fatigue_Diff",
        # Short-term (streak) differentials
        "OPS_Diff_S",
        "K_Rate_Diff_S",
        "SP_K9_Diff_S",
        # Run differential (team quality)
        "RunDiff_Diff",
        # Rest
        "Rest_Diff",
        "SP_Rest_Diff",
        # Home field advantage signal
        "Home_Win_Rate",
    ]

    print(f"Training on {len(features)} Features: {features}")

    # Chronological split: pre-2026 = train+val, 2026 = test
    pre2026 = df[df["Date"].dt.year < 2026].copy()
    test_df = df[df["Date"].dt.year >= 2026].copy()

    # Carve out last 10% of pre-2026 data as a held-out validation set
    # for early stopping — this prevents test-set leakage entirely
    val_cutoff = int(len(pre2026) * 0.90)
    train_df = pre2026.iloc[:val_cutoff]
    val_df = pre2026.iloc[val_cutoff:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"\nSplit sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\nTraining XGBoost model...")
    # Hyperparameters tuned via tune.py TimeSeriesSplit grid search
    model = XGBClassifier(
        n_estimators=1000,       # high ceiling — early stopping will find the right number
        learning_rate=0.01,      # tuned
        max_depth=3,             # tuned — shallow trees reduce overfit on small dataset
        subsample=0.7,           # tuned
        colsample_bytree=0.7,    # tuned
        min_child_weight=3,      # tuned
        gamma=1.0,               # tuned — minimum loss reduction to make a split
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=42,
    )

    # Early stopping uses the validation set (NOT the test set)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"Early stopping triggered! Model stopped at tree #{model.best_iteration}")

    # Re-train on train+val combined using the best iteration count found by early stopping
    # This gives the model more data without touching the test set
    print(f"Re-fitting on train+val ({len(X_train)+len(X_val)} games) at best_iteration={model.best_iteration}...")
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model = XGBClassifier(
        n_estimators=model.best_iteration,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    final_model.fit(X_trainval, y_trainval)

    # ==========================================
    # EVALUATE & GENERATE BETTING TICKET
    # ==========================================
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred_proba)

    # Val set performance for reference (using early-stopping model before re-fit)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_ll = log_loss(y_val, y_val_proba)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Val  Accuracy: {val_acc:.2%}  |  Val  Log Loss: {val_ll:.4f}")
    print(f"Test Accuracy: {acc:.2%}  |  Test Log Loss: {ll:.4f}")

    # ---------------------------------------------------------
    # THE BETTING TICKET
    # ---------------------------------------------------------
    results_df = test_df[["Date", "Away_Team", "Home_Team", "Home_Win"]].copy()
    results_df["Home_Prob"] = y_pred_proba
    results_df["Away_Prob"] = 1 - y_pred_proba
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
        {"Feature": features, "Importance": final_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    print(importance.to_string(index=False))

    # Confidence distribution sanity check
    print("\nPrediction confidence distribution (test set):")
    bins = [0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
    labels = ["<45%", "45-50%", "50-55%", "55-60%", "60-65%", ">65%"]
    home_probs = pd.cut(results_df["Home_Prob"], bins=bins, labels=labels)
    print(home_probs.value_counts().sort_index())

    # ==========================================
    # SAVE MODEL & METADATA
    # ==========================================
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    meta = {
        "features": features,
        "feature_medians": {col: float(df[col].median()) for col in features},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model metadata saved to {META_PATH}")

    return model


if __name__ == "__main__":
    train_baseline_model("data/mlb_model_ready.csv")
