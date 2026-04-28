# MLB Game Outcome Predictor (XGBoost)

End-to-end pipeline for predicting today's and tomorrow's MLB game winners using XGBoost. Pulls free data from `pybaseball` (Statcast/Baseball Reference/FanGraphs) and `MLB-StatsAPI` (real-time schedule + probable pitchers).

## Setup

```bash
pip install -r requirements.txt
```

## One-time backfill

Pulls last season + this season-to-date games + season-level team and pitcher stats. Takes ~5-8 minutes. Be polite — Baseball Reference rate-limits aggressive scrapers.

If you want more (or less) data, edit `BACKFILL_START_YEAR` in `scripts/config.py`.

```bash
python scripts/backfill_history.py
```

## Build features + train

```bash
python scripts/build_features.py
python scripts/train_model.py
```

## Daily run (schedule via cron)

```bash
python scripts/daily_update.py     # 7am: append yesterday + pull next 3 days schedule
python scripts/build_features.py   # rebuild feature matrix
python scripts/predict_today.py    # write today's predictions
```

A reasonable cron entry (run at 7:00 AM daily):
```
0 7 * * * cd /path/to/mlb_predictor && python scripts/daily_update.py && python scripts/predict_today.py
```

You don't strictly need to **retrain** every day — once a week is fine in-season. Just re-run `daily_update.py` + `predict_today.py` to get fresh predictions using the current model.

## What's in the feature set

Per game, for both home and away teams:

- **Recent form** (last 15 games): win pct, runs scored/allowed, run differential
- **Season-to-date**: win pct (excluding current game)
- **Rest**: days since last game, and diff vs. opponent
- **Probable starting pitcher**: ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9, WAR (current season, falls back to prior season early in the year)
- **Differentials**: home minus away on the most predictive metrics

Target: `home_win` (1 if home team won, 0 otherwise).

## Realistic expectations

The Vegas line on baseball games clears ~58-60% accuracy. A solid model with the features above and ~2 seasons of training data lands around **53-57% out-of-sample accuracy** — that's genuinely good (random = 50%, naive "always pick home" = ~52-54%). What matters more than raw accuracy is **calibration** (your 70% predictions should win 70% of the time) — that's why `train_model.py` reports log loss and Brier score.

Early in the current season (April/May), expect predictions to be noisier — rolling form features need at least a few games per team to stabilize. The `dropna()` in `build_features.py` removes rows that don't have enough history yet.

## Things to improve next

These are what would push you from 56% → 58%+:

1. **Bullpen quality** — pitcher stats here cover the starter only. Add team bullpen ERA over the rolling window. Pull from `pybaseball.team_pitching` filtered to relievers.
2. **Park factors** — Coors Field is wildly different from Petco. Pull `pybaseball.park_factors()` and join on the home venue.
3. **Lineup strength** — Use `MLB-StatsAPI` `boxscore` to get the actual posted lineup pre-game and aggregate batter wOBA/wRC+ from `pybaseball.batting_stats`.
4. **Travel/timezone** — Add a feature for time zones crossed since the last game.
5. **Pitcher-vs-batter splits** — Available in Statcast but expensive to compute; only worth it if you're pushing for the last 1-2% of accuracy.
6. **Weather** — Wind direction + temperature at outdoor parks affects scoring.
7. **Model calibration** — Wrap the XGBoost output in a `CalibratedClassifierCV` (sigmoid or isotonic). Critical if you want to use these probabilities for any kind of betting decision.

## File layout

```
mlb_predictor/
├── data/
│   ├── raw/              # pulled-from-source data (parquet)
│   └── processed/        # feature matrix + predictions
├── models/               # trained model artifacts
├── scripts/
│   ├── config.py
│   ├── backfill_history.py
│   ├── daily_update.py
│   ├── build_features.py
│   ├── train_model.py
│   └── predict_today.py
└── requirements.txt
```

## Troubleshooting

- **Rate-limited / 429 from Baseball Reference**: `backfill_history.py` already sleeps between requests. If you still get blocked, lengthen `time.sleep(0.3)` to `time.sleep(1.0)`.
- **Team code mismatches**: `pybaseball` uses Baseball Reference codes (CHW, KCR, SDP, SFG, TBR, WSN). MLB Stats API uses full team names. The mapping in `daily_update.py:TEAM_NAME_TO_CODE` handles this.
- **Pitcher names don't join**: FanGraphs uses "Shohei Ohtani"; the MLB Stats API gives the same. If you see lots of NaN pitcher features, `print(df[df["home_p_era"].isna()]["home_probable_pitcher"].unique())` and add an alias map.
- **NaN explosion early in the season**: by design — rolling features need 3+ games per team. The `dropna()` in `build_features.py` removes the unusable early rows from training; for prediction the snapshot uses whatever's available.

## Ethics / disclaimer

This is a personal project for fun and learning. Sports betting is statistically a losing proposition for almost everyone — sportsbook lines incorporate far more information than this model. Don't bet money you can't afford to lose, and check that sports betting is legal in your jurisdiction.
