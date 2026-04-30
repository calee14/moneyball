"""
Ablation study for the Elo engine.

Runs four configurations on the same data:
    (1) Pure team Elo    : team rating + home field, no starter, no travel
    (2) + Starter delta  : adds Game Score-based starter ratings
    (3) + Travel/Rest    : adds penalty function
    (4) Full model       : everything

Evaluates each on full-season 2025 and partial 2026 holdouts so we can see
whether the ceiling is the model or the test set.

Usage: python ablate.py data/mlb_games_with_features.csv
"""

import sys
import math
import pandas as pd
from copy import deepcopy

import elo_engine
from elo_engine import EloEngine, attach_penalties


def run_config(
    df,
    holdout_seasons,
    use_starter=True,
    use_travel=True,
    K=elo_engine.K_FACTOR,
    gs2_mult=elo_engine.GS2_TO_ELO,
    sp_cap=elo_engine.SP_DELTA_CAP,
    hfa=elo_engine.HOME_FIELD_ADV,
):
    """Run engine with selected features. Returns metrics per holdout."""
    # Patch constants
    elo_engine.K_FACTOR = K
    elo_engine.GS2_TO_ELO = gs2_mult
    elo_engine.SP_DELTA_CAP = sp_cap
    elo_engine.HOME_FIELD_ADV = hfa

    # Mutate the data per config — this is the cleanest way to disable features
    # without modifying the engine itself
    work = df.copy()
    if not use_starter:
        # Null out starter IDs so starter_delta returns 0
        work["Home_SP_ID"] = None
        work["Away_SP_ID"] = None
    if not use_travel:
        work["Home_rest_travel_penalty"] = 0.0
        work["Away_rest_travel_penalty"] = 0.0

    engine = EloEngine()
    preds = []
    for _, row in work.iterrows():
        preds.append(engine.process_game(row))
    pred_df = pd.DataFrame(preds)

    results = {}
    for season in holdout_seasons:
        held = pred_df[pred_df["Season"] == season]
        if len(held) <= 30:
            continue
        held = held.iloc[30:]  # post-warmup
        p = held["p_home_win"].clip(1e-6, 1 - 1e-6)
        y = held["actual_home_win"]
        logloss = -(y * (p.map(math.log)) + (1 - y) * ((1 - p).map(math.log))).mean()
        brier = ((p - y) ** 2).mean()
        acc = ((p > 0.5).astype(int) == y).mean()
        results[season] = {
            "n": len(held),
            "logloss": logloss,
            "brier": brier,
            "acc": acc,
        }

    ratings = list(engine.team_elo.values())
    spread = max(ratings) - min(ratings) if ratings else 0
    return results, spread


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/mlb_games_with_features.csv"
    print(f"Loading {in_path}...")
    df = pd.read_csv(in_path)
    df = attach_penalties(df)
    if "Game_DateTime" in df.columns:
        df["_sort"] = pd.to_datetime(df["Game_DateTime"], errors="coerce", utc=True)
    else:
        df["_sort"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["_sort", "Game_Number"]).reset_index(drop=True)
    print(f"  {len(df)} games loaded\n")

    holdouts = [2025, 2026]
    configs = [
        ("(1) Pure team Elo", dict(use_starter=False, use_travel=False)),
        ("(2) + Starter delta", dict(use_starter=True, use_travel=False)),
        ("(3) + Travel/Rest", dict(use_starter=False, use_travel=True)),
        ("(4) Full model", dict(use_starter=True, use_travel=True)),
    ]

    print(
        f"{'Config':<24} {'Holdout':<8} {'n':<6} {'logloss':<10} "
        f"{'brier':<10} {'acc':<8} {'spread':<8}"
    )
    print("-" * 80)

    all_results = []
    for label, cfg in configs:
        results, spread = run_config(df, holdouts, **cfg)
        for season, m in results.items():
            print(
                f"{label:<24} {season:<8} {m['n']:<6} "
                f"{m['logloss']:<10.4f} {m['brier']:<10.4f} "
                f"{m['acc']:<8.4f} {spread:<8.0f}"
            )
            all_results.append(
                {"config": label, "season": season, "rating_spread": spread, **m}
            )
        print()

    # Compute lift of each feature
    res_df = pd.DataFrame(all_results)
    print("\n=== Feature lift (logloss change vs pure team Elo) ===")
    for season in holdouts:
        sub = res_df[res_df.season == season].set_index("config")
        if "(1) Pure team Elo" not in sub.index:
            continue
        baseline = sub.loc["(1) Pure team Elo", "logloss"]
        print(f"\n  {season} (baseline pure-Elo logloss = {baseline:.4f}):")
        for cfg in ["(2) + Starter delta", "(3) + Travel/Rest", "(4) Full model"]:
            if cfg in sub.index:
                lift = sub.loc[cfg, "logloss"] - baseline
                arrow = "↓ better" if lift < 0 else "↑ worse "
                print(f"    {cfg:<22}  {lift:+.4f}  {arrow}")

    # Reference baselines for context
    print("\n=== Reference baselines ===")
    print(f"  Always 50%:    logloss = 0.6931")
    print(f"  Always 54%:    logloss = 0.6899  (matches MLB avg home win rate)")

    res_df.to_csv("data/ablation_results.csv", index=False)
    print("\nFull results -> data/ablation_results.csv")


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    main()
