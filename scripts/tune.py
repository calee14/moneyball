"""
Hyperparameter sweep for the Elo engine.

For each combination of (K, GS2_TO_ELO, SP_DELTA_CAP, HOME_FIELD_ADV),
runs the full backtest and records OOS logloss/brier/accuracy.

Best combo by OOS logloss is the one to use in production.

Usage:
    python tune.py data/mlb_games_with_features.csv 2026
"""

import sys
import itertools
import math
import pandas as pd

import elo_engine
from elo_engine import EloEngine, attach_penalties


def run_one(df, holdout_season, K, gs2_mult, sp_cap, hfa):
    """Train + evaluate with one parameter combo. Returns metrics dict."""
    # Patch module-level constants. Hacky but contained.
    elo_engine.K_FACTOR = K
    elo_engine.GS2_TO_ELO = gs2_mult
    elo_engine.SP_DELTA_CAP = sp_cap
    elo_engine.HOME_FIELD_ADV = hfa

    engine = EloEngine()
    preds = []
    for _, row in df.iterrows():
        preds.append(engine.process_game(row))
    pred_df = pd.DataFrame(preds)

    # Evaluate on holdout, post-warmup (first 30 games skipped)
    held = pred_df[pred_df["Season"] == holdout_season]
    if len(held) <= 30:
        return None
    held = held.iloc[30:]
    p = held["p_home_win"].clip(1e-6, 1 - 1e-6)
    y = held["actual_home_win"]
    logloss = -(y * (p.map(math.log)) + (1 - y) * ((1 - p).map(math.log))).mean()
    brier = ((p - y) ** 2).mean()
    acc = ((p > 0.5).astype(int) == y).mean()

    # Also report rating spread — bad sign if it's tiny
    ratings = list(engine.team_elo.values())
    spread = max(ratings) - min(ratings) if ratings else 0

    return {
        "K": K,
        "gs2_mult": gs2_mult,
        "sp_cap": sp_cap,
        "hfa": hfa,
        "logloss": logloss,
        "brier": brier,
        "acc": acc,
        "rating_spread": spread,
        "n": len(held),
    }


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/mlb_games_with_features.csv"
    holdout = int(sys.argv[2]) if len(sys.argv) > 2 else 2026

    print(f"Loading {in_path}...")
    df = pd.read_csv(in_path)
    df = attach_penalties(df)
    if "Game_DateTime" in df.columns:
        df["_sort"] = pd.to_datetime(df["Game_DateTime"], errors="coerce", utc=True)
    else:
        df["_sort"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["_sort", "Game_Number"]).reset_index(drop=True)
    print(f"  {len(df)} games, holdout={holdout}\n")

    # Grid. Keep it modest — 81 combos is plenty for a first pass.
    K_grid = [3, 5, 7, 9]
    gs2_grid = [1.0, 2.0, 3.0]
    cap_grid = [40, 60, 80]
    hfa_grid = [18, 24, 30]

    combos = list(itertools.product(K_grid, gs2_grid, cap_grid, hfa_grid))
    print(f"Sweeping {len(combos)} combinations...\n")

    results = []
    for i, (K, gs2, cap, hfa) in enumerate(combos, 1):
        r = run_one(df, holdout, K, gs2, cap, hfa)
        if r:
            results.append(r)
        if i % 10 == 0:
            print(f"  {i}/{len(combos)} done")

    res_df = pd.DataFrame(results).sort_values("logloss")
    print("\n=== Top 10 (lowest OOS logloss) ===")
    print(res_df.head(10).to_string(index=False))
    print("\n=== Bottom 5 (worst) ===")
    print(res_df.tail(5).to_string(index=False))

    # Marginal effect of each param: avg logloss when fixed at each value
    print("\n=== Marginal effects (avg logloss when this param = value) ===")
    for col in ["K", "gs2_mult", "sp_cap", "hfa"]:
        marg = res_df.groupby(col)["logloss"].mean().round(4)
        print(f"\n  {col}:")
        for k, v in marg.items():
            print(f"    {k}: {v}")

    res_df.to_csv("data/tune_results.csv", index=False)
    print("\nFull results -> data/tune_results.csv")

    best = res_df.iloc[0]
    print(f"\n=== Best combo ===")
    print(
        f"  K={int(best.K)}  GS2_TO_ELO={best.gs2_mult}  "
        f"SP_DELTA_CAP={int(best.sp_cap)}  HOME_FIELD_ADV={int(best.hfa)}"
    )
    print(
        f"  -> logloss={best.logloss:.4f}  acc={best.acc:.4f}  "
        f"rating_spread={best.rating_spread:.0f}"
    )


if __name__ == "__main__":
    main()

