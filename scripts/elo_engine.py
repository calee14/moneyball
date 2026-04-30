"""
Elo engine for MLB.

Trains by walking games chronologically:
  - team Elo updates after each game (K=4, low because of high variance sport)
  - starter "delta" updates after each start, derived from rolling Game Score 2.0
  - between seasons, ratings regress 1/3 of the way to 1500

Win probability for a future game:
    diff = (home_team_elo + home_sp_delta)
         - (away_team_elo + away_sp_delta)
         + HOME_FIELD_ADV
         - home_rest_travel_penalty
         + away_rest_travel_penalty
    P(home win) = 1 / (1 + 10^(-diff / 400))

Backtest reports log-loss, Brier score, and accuracy on held-out games.
"""

import math
import json
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime

from travel_rest import (
    build_team_timeline,
    add_rest_travel_features,
    merge_back,
    rest_travel_penalty,
    _ip_to_float,
)


# ---------------------------------------------------------------------------
# Constants — starting points. Tune empirically once you have a backtest.
# ---------------------------------------------------------------------------
INITIAL_ELO = 1500
K_FACTOR = 6  # team Elo learning rate (low; baseball is noisy)
HOME_FIELD_ADV = 24  # ~54% baseline home win rate
SEASON_REGRESSION = 1 / 3  # fraction of way back to 1500 each offseason

# Starter delta: convert Game Score 2.0 to Elo points
GS2_LEAGUE_AVG = 50  # GS2 of an average start
GS2_TO_ELO = 2.0  # 1 GS2 point above avg = 2 Elo points (capped)
SP_DELTA_CAP = 60  # max abs starter delta in Elo points
SP_ROLLING_STARTS = 10  # rolling window for a starter's recent form
SP_PRIOR_WEIGHT = 3  # shrink toward league avg when sample is small


# ---------------------------------------------------------------------------
# Game Score 2.0 — Tom Tango's variant, uses K/BB/HR/R/outs (what we collect)
# ---------------------------------------------------------------------------
def game_score_2(ip_str, k, bb, hr, er):
    """GS2 = 40 + 2*outs + K - 2*BB - 3*HR - 2*R. We use ER as a proxy for R."""
    ip = _ip_to_float(ip_str)
    outs = round(ip * 3)
    return 40 + 2 * outs + k - 2 * bb - 3 * hr - 2 * er


# ---------------------------------------------------------------------------
# Win probability
# ---------------------------------------------------------------------------
def win_probability(
    home_elo,
    away_elo,
    home_sp_delta=0.0,
    away_sp_delta=0.0,
    home_penalty=0.0,
    away_penalty=0.0,
):
    """Probability the home team wins."""
    diff = (
        (home_elo + home_sp_delta)
        - (away_elo + away_sp_delta)
        + HOME_FIELD_ADV
        - home_penalty
        + away_penalty
    )
    return 1.0 / (1.0 + 10 ** (-diff / 400))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class EloEngine:
    def __init__(self):
        self.team_elo = defaultdict(lambda: INITIAL_ELO)
        # rolling deque of recent GS2 per starter
        self.sp_history = defaultdict(lambda: deque(maxlen=SP_ROLLING_STARTS))
        self.last_season_seen = None

    # -- starter delta ------------------------------------------------------
    def starter_delta(self, sp_id):
        """Return Elo-point delta for this starter based on rolling GS2."""
        if sp_id is None or pd.isna(sp_id):
            return 0.0
        history = self.sp_history[sp_id]
        n = len(history)
        if n == 0:
            return 0.0
        # Bayesian shrinkage: weighted avg of starter's GS2 and league mean
        avg_gs = sum(history) / n
        shrunk = (n * avg_gs + SP_PRIOR_WEIGHT * GS2_LEAGUE_AVG) / (n + SP_PRIOR_WEIGHT)
        delta = (shrunk - GS2_LEAGUE_AVG) * GS2_TO_ELO
        return max(-SP_DELTA_CAP, min(SP_DELTA_CAP, delta))

    # -- season rollover ----------------------------------------------------
    def _maybe_regress(self, season):
        """Regress all ratings toward 1500 when we cross a season boundary."""
        if self.last_season_seen is None:
            self.last_season_seen = season
            return
        if season != self.last_season_seen:
            for team in list(self.team_elo.keys()):
                cur = self.team_elo[team]
                self.team_elo[team] = cur + SEASON_REGRESSION * (INITIAL_ELO - cur)
            # starter histories: clear them. Pitcher form doesn't carry across
            # a 5-month layoff in any meaningful way, and many pitchers will
            # have had surgery, role changes, etc.
            self.sp_history.clear()
            self.last_season_seen = season

    # -- main per-game step -------------------------------------------------
    def process_game(self, row):
        """
        Process one game row. Returns dict with pre-game predictions; mutates
        engine state (team Elo + starter histories) using the actual outcome.
        """
        season = pd.to_datetime(row["Date"]).year
        self._maybe_regress(season)

        home_id = row["Home_Team_ID"]
        away_id = row["Away_Team_ID"]
        home_sp = row.get("Home_SP_ID")
        away_sp = row.get("Away_SP_ID")

        # --- pre-game state ---
        home_elo_pre = self.team_elo[home_id]
        away_elo_pre = self.team_elo[away_id]
        home_sp_d = self.starter_delta(home_sp)
        away_sp_d = self.starter_delta(away_sp)
        home_pen = row.get("Home_rest_travel_penalty", 0.0) or 0.0
        away_pen = row.get("Away_rest_travel_penalty", 0.0) or 0.0

        p_home = win_probability(
            home_elo_pre,
            away_elo_pre,
            home_sp_d,
            away_sp_d,
            home_pen,
            away_pen,
        )

        # --- update team Elo on actual result, scaled by margin of victory ---
        actual = row["Home_Win"]
        run_diff = abs(row["Home_Score"] - row["Away_Score"])
        # Higher Elo team's pre-game advantage (signed by who won)
        if actual == 1:
            elo_diff_winner = (home_elo_pre + home_sp_d) - (away_elo_pre + away_sp_d)
        else:
            elo_diff_winner = (away_elo_pre + away_sp_d) - (home_elo_pre + home_sp_d)
        # FiveThirtyEight-style MOV multiplier: log scaling + autocorrelation correction
        mov_mult = math.log(max(run_diff, 1) + 1) * (
            2.2 / (elo_diff_winner * 0.001 + 2.2)
        )
        delta = K_FACTOR * mov_mult * (actual - p_home)
        self.team_elo[home_id] = home_elo_pre + delta
        self.team_elo[away_id] = away_elo_pre - delta

        # --- update starter histories with this game's GS2 ---
        if pd.notna(home_sp):
            gs = game_score_2(
                row["Home_SP_IP"],
                row["Home_SP_K"],
                row["Home_SP_BB"],
                row["Home_SP_HR"],
                row["Home_SP_ER"],
            )
            self.sp_history[home_sp].append(gs)
        if pd.notna(away_sp):
            gs = game_score_2(
                row["Away_SP_IP"],
                row["Away_SP_K"],
                row["Away_SP_BB"],
                row["Away_SP_HR"],
                row["Away_SP_ER"],
            )
            self.sp_history[away_sp].append(gs)

        return {
            "Game_ID": row["Game_ID"],
            "Date": row["Date"],
            "Season": season,
            "home_team": row["Home_Team"],
            "away_team": row["Away_Team"],
            "home_elo_pre": home_elo_pre,
            "away_elo_pre": away_elo_pre,
            "home_sp_delta": home_sp_d,
            "away_sp_delta": away_sp_d,
            "home_penalty": home_pen,
            "away_penalty": away_pen,
            "p_home_win": p_home,
            "actual_home_win": actual,
        }

    # -- snapshot for live use ---------------------------------------------
    def snapshot(self):
        """Return current ratings as JSON-serializable dict."""
        return {
            "team_elo": dict(self.team_elo),
            "sp_recent_gs2": {sp: list(h) for sp, h in self.sp_history.items()},
            "last_season_seen": self.last_season_seen,
        }

    def load_snapshot(self, snap):
        self.team_elo = defaultdict(lambda: INITIAL_ELO, snap["team_elo"])
        self.sp_history = defaultdict(lambda: deque(maxlen=SP_ROLLING_STARTS))
        for sp, hist in snap["sp_recent_gs2"].items():
            self.sp_history[sp] = deque(hist, maxlen=SP_ROLLING_STARTS)
        self.last_season_seen = snap["last_season_seen"]


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------
def attach_penalties(df_with_features):
    """Compute per-game rest/travel penalty for both teams from feature cols."""
    df = df_with_features.copy()

    def pen(row, side):
        return rest_travel_penalty(
            row.get(f"{side}_days_rest"),
            row.get(f"{side}_travel_miles"),
            row.get(f"{side}_is_day_after_night", False),
            row.get(f"{side}_is_doubleheader_g2", False),
            row.get(f"{side}_bullpen_ip_last3"),
        )

    df["Away_rest_travel_penalty"] = df.apply(lambda r: pen(r, "Away"), axis=1)
    df["Home_rest_travel_penalty"] = df.apply(lambda r: pen(r, "Home"), axis=1)
    return df


def evaluate(predictions_df, label="all"):
    """Log-loss, Brier, accuracy. Reference: a 54% home prior gives logloss ~0.687."""
    p = predictions_df["p_home_win"].clip(1e-6, 1 - 1e-6)
    y = predictions_df["actual_home_win"]
    logloss = -(y * (p.map(math.log)) + (1 - y) * ((1 - p).map(math.log))).mean()
    brier = ((p - y) ** 2).mean()
    acc = ((p > 0.5).astype(int) == y).mean()
    print(
        f"  [{label:>10}] n={len(predictions_df):>5}  "
        f"logloss={logloss:.4f}  brier={brier:.4f}  acc={acc:.4f}"
    )
    return {"n": len(predictions_df), "logloss": logloss, "brier": brier, "acc": acc}


def calibration_report(predictions_df, label="all", n_bins=10):
    """Bucket by predicted probability and compare to actual win rate.
    Well-calibrated -> predicted ~= actual in every bucket."""
    df = predictions_df.copy()
    df["bin"] = pd.cut(
        df["p_home_win"],
        bins=n_bins,
        labels=[f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}" for i in range(n_bins)],
        include_lowest=True,
    )
    grouped = df.groupby("bin", observed=True).agg(
        n=("actual_home_win", "size"),
        predicted=("p_home_win", "mean"),
        actual=("actual_home_win", "mean"),
    )
    grouped["gap"] = (grouped["actual"] - grouped["predicted"]).round(3)
    print(f"\n  Calibration [{label}]:")
    print(grouped.to_string())


def run_backtest(
    features_csv,
    predictions_out="data/predictions.csv",
    ratings_out="data/ratings_snapshot.json",
    holdout_season=None,
):
    """
    Walk all games chronologically, predicting then updating.
    If holdout_season is given, evaluate that season separately as out-of-sample
    (its predictions still come from a model that has only seen prior games,
    since we predict before updating).
    """
    print(f"Loading {features_csv}...")
    df = pd.read_csv(features_csv)
    df = attach_penalties(df)

    # Sort chronologically (datetime if available, else date + game number)
    if "Game_DateTime" in df.columns:
        df["_sort"] = pd.to_datetime(df["Game_DateTime"], errors="coerce", utc=True)
    else:
        df["_sort"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["_sort", "Game_Number"]).reset_index(drop=True)

    print(f"Walking {len(df)} games...")
    engine = EloEngine()
    preds = []
    for _, row in df.iterrows():
        preds.append(engine.process_game(row))

    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(predictions_out, index=False)
    print(f"Predictions -> {predictions_out}")

    with open(ratings_out, "w") as f:
        json.dump(engine.snapshot(), f, indent=2, default=str)
    print(f"Ratings snapshot -> {ratings_out}")

    print("\n=== Backtest results ===")
    evaluate(pred_df, "all")
    for season, sub in pred_df.groupby("Season"):
        evaluate(sub, str(season))
    if holdout_season is not None:
        held = pred_df[pred_df["Season"] == holdout_season]
        if len(held):
            print(f"\n=== Out-of-sample ({holdout_season}) ===")
            evaluate(held, f"OOS {holdout_season}")
            # Skip first 30 games of season — gives the model a few series to
            # adjust to the regressed starting ratings
            evaluate(held.iloc[30:], f"OOS {holdout_season} (post-warmup)")
            calibration_report(held, f"OOS {holdout_season}")
    else:
        calibration_report(pred_df, "all")

    return engine, pred_df


# ---------------------------------------------------------------------------
# Predict a future / upcoming game from a saved snapshot
# ---------------------------------------------------------------------------
def predict_game(
    engine,
    home_team_id,
    away_team_id,
    home_sp_id=None,
    away_sp_id=None,
    home_penalty=0.0,
    away_penalty=0.0,
):
    """Return win probability dict for a hypothetical/upcoming matchup."""
    h_elo = engine.team_elo[home_team_id]
    a_elo = engine.team_elo[away_team_id]
    h_sp = engine.starter_delta(home_sp_id)
    a_sp = engine.starter_delta(away_sp_id)
    p = win_probability(h_elo, a_elo, h_sp, a_sp, home_penalty, away_penalty)
    return {
        "home_elo": h_elo,
        "away_elo": a_elo,
        "home_sp_delta": h_sp,
        "away_sp_delta": a_sp,
        "home_penalty": home_penalty,
        "away_penalty": away_penalty,
        "p_home_win": p,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/mlb_games_with_features.csv"
    holdout = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_backtest(in_path, holdout_season=holdout)
