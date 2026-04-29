import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Import the shared feature builder so tune and train always stay in sync
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import build_features


def tune_xgboost(filepath):
    print("Loading model-ready data...")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = build_features(df)

    features = [
        "Park_Factor",
        "OPS_Diff",
        "K_Rate_Diff",
        "Bullpen_ERA_Diff",
        "SP_FIP_Diff",
        "SP_K9_Diff",
        "Bullpen_Fatigue_Diff",
        "OPS_Diff_S",
        "K_Rate_Diff_S",
        "SP_K9_Diff_S",
        "RunDiff_Diff",
        "Rest_Diff",
        "SP_Rest_Diff",
        "Home_Win_Rate",
    ]
    target = "Home_Win"

    # Only tune on pre-2026 data
    train_df = df[df["Date"].dt.year < 2026].copy()
    X_train = train_df[features]
    y_train = train_df[target]

    print(f"Starting Grid Search on {len(X_train)} historical games...")

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05],
        "n_estimators": [200, 400, 600],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [3, 5, 8],
        "gamma": [0, 0.5, 1.0],
    }

    # TimeSeriesSplit ensures no future-data leakage during cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    model = XGBClassifier(random_state=42, eval_metric="logloss")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=tscv,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 40)
    print("BEST XGBOOST SETTINGS FOUND")
    print("=" * 40)

    best_params = grid_search.best_params_
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print(f"\nBest Historical Log Loss: {-grid_search.best_score_:.4f}")
    print("\nCopy these settings into your train.py script!")


if __name__ == "__main__":
    tune_xgboost("data/mlb_model_ready.csv")
