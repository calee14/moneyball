# MLB Elo Predictor

Pure team-Elo predictor for MLB games, with a TUI for browsing today's slate and comparing teams side-by-side.

## Setup

```bash
pip install -r requirements.txt
```

## First run

Pull historical games (~10 min), build features, train ratings:

```bash
python scripts/collect_data.py
python scripts/travel_rest.py
python scripts/elo_engine.py data/mlb_games_with_features.csv
```

## Daily use

```bash
python scripts/tui.py
```

Inside the TUI:

| Key       | Action                          |
|-----------|---------------------------------|
| `↑` `↓`   | Navigate games                  |
| `Enter` `A` | Show Elo analysis             |
| `R`       | Refresh data + ratings (~30s)   |
| `T`       | Cycle filter (All/Today/Tomorrow) |
| `Q`       | Quit                            |

## What you see per game

- Win probability (pure team Elo + home field)
- Side-by-side comparison: Elo, last 10 record, run diff, rest, bullpen IP, starter form
- Flags: hot/cold streaks, bullpen burnout, stale data warnings

## How it works

Elo updates after each game with K=6, MOV-scaled, regressed 1/3 toward 1500 between seasons. Win probability uses team Elo + 24-point home field advantage. Starter form and rest/travel are surfaced as context, not folded into the model.

## Honest expectations

Logloss ~0.68, accuracy ~55%. Beats "always pick home" but not by much. Vegas does better. The TUI exists so you can apply gut feel on top of the number — that's where the value is.

## Files

```
scripts/
├── collect_data.py     # pull box scores from MLB StatsAPI
├── travel_rest.py      # add rest, travel, bullpen features
├── elo_engine.py       # walk games, build ratings, backtest
├── elo_predictor.py    # win prob + team context for the TUI
├── refresh.py          # incremental refresh (used by TUI's R key)
├── tui.py              # the app
├── tune.py             # hyperparameter sweep
└── ablate.py           # feature ablation study
data/
├── mlb_historical_games.csv
├── mlb_games_with_features.csv
└── ratings_snapshot.json
```

## Disclaimer

Personal project. Don't bet on it. Vegas knows more than this model does.
