"""
refresh.py — incremental data + Elo refresh.

Reads the existing CSV, finds the last date in it, fetches only games
since then (plus a 1-day safety overlap), appends, re-runs feature
engineering and Elo, and writes new snapshot.

Designed to be called from the TUI — yields progress strings so the UI
can show live status.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Iterable

import pandas as pd
import requests

# Reuse what's already there
from collect_data import HEADER, fetch_game
from travel_rest import (
    add_rest_travel_features,
    build_team_timeline,
    merge_back,
)
from elo_engine import run_backtest


RAW_CSV = Path("data/mlb_historical_games.csv")
FEATURES_CSV = Path("data/mlb_games_with_features.csv")
RATINGS_JSON = Path("data/ratings_snapshot.json")

session = requests.Session()


# ── helpers ───────────────────────────────────────────────────────────────────
def _last_date_in_csv(csv_path: Path) -> date | None:
    """Return the most recent date in the CSV, or None if empty/missing."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["Date"])
        if df.empty:
            return None
        return pd.to_datetime(df["Date"]).max().date()
    except Exception:
        return None


def _existing_game_ids(csv_path: Path) -> set[int]:
    """Return all Game_IDs already in the CSV — used to dedupe."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["Game_ID"])
        return set(int(g) for g in df["Game_ID"].dropna())
    except Exception:
        return set()


def _fetch_completed_game_ids_in_range(start: date, end: date) -> list[int]:
    """Schedule-API call for completed regular-season games in [start, end]."""
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
        f"&startDate={start.isoformat()}&endDate={end.isoformat()}&gameType=R"
    )
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return []
    data = r.json()
    ids = []
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            if g.get("status", {}).get("statusCode") in ("F", "O"):
                ids.append(int(g["gamePk"]))
    return ids


# ── main pipeline ─────────────────────────────────────────────────────────────
def refresh(
    progress: callable = print,
    full: bool = False,
    seasons_for_full: Iterable[int] = (2024, 2025, 2026),
    max_workers: int = 15,
) -> dict:
    """
    Refresh data and ratings.

    progress: callable(str) — called with status strings as work proceeds.
    full: if True, re-fetch every season. Otherwise incremental.

    Returns: dict with {"new_games": N, "total_games": M, "ok": True}.
    """
    os.makedirs("data", exist_ok=True)

    # ── 1. Decide what date range to fetch ────────────────────────────────────
    if full or not RAW_CSV.exists():
        progress("Full refresh — fetching all seasons...")
        all_ids = []
        for year in seasons_for_full:
            progress(f"  Fetching schedule for {year}...")
            all_ids.extend(
                _fetch_completed_game_ids_in_range(date(year, 3, 20), date(year, 11, 5))
            )
        seen = set()
        new_ids = [g for g in all_ids if not (g in seen or seen.add(g))]
        existing = set()
    else:
        last = _last_date_in_csv(RAW_CSV)
        # Overlap by 2 days to catch games that were live last time and are
        # now final, plus any late-arriving box scores
        if last is None:
            start = date.today() - timedelta(days=7)
        else:
            start = last - timedelta(days=2)
        end = date.today()
        progress(f"Incremental refresh — fetching {start} to {end}...")
        candidate_ids = _fetch_completed_game_ids_in_range(start, end)
        existing = _existing_game_ids(RAW_CSV)
        new_ids = [g for g in candidate_ids if g not in existing]
        progress(f"  {len(candidate_ids)} games in window, {len(new_ids)} new vs CSV.")

    # ── 2. Fetch the new games' box scores ────────────────────────────────────
    if not new_ids:
        progress("No new games to fetch.")
    else:
        progress(f"Fetching {len(new_ids)} game box scores...")
        rows = []
        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_game, pk): pk for pk in new_ids}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                if row is not None:
                    rows.append(row)
                    ok += 1
                else:
                    fail += 1
                if i % 25 == 0 or i == len(new_ids):
                    progress(f"  {i}/{len(new_ids)} (ok={ok}, fail={fail})")

        # Append (or write fresh on full refresh)
        write_header = full or not RAW_CSV.exists()
        mode = "w" if write_header else "a"
        with open(RAW_CSV, mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(HEADER)
            for r in rows:
                w.writerow(r)
        progress(f"Appended {ok} games to {RAW_CSV}")

    # ── 3. Re-run feature engineering on the full CSV ─────────────────────────
    progress("Recomputing rest/travel features...")
    df = pd.read_csv(RAW_CSV)
    timeline = build_team_timeline(df)
    timeline = add_rest_travel_features(timeline)
    out = merge_back(df, timeline)
    out.to_csv(FEATURES_CSV, index=False)
    progress(f"  -> {FEATURES_CSV} ({len(out)} games)")

    # ── 4. Rebuild Elo snapshot from scratch ──────────────────────────────────
    progress("Rebuilding Elo ratings...")
    # run_backtest writes ratings_snapshot.json + predictions.csv
    run_backtest(
        str(FEATURES_CSV),
        predictions_out="data/predictions.csv",
        ratings_out=str(RATINGS_JSON),
    )
    progress("Elo ratings rebuilt.")

    return {
        "new_games": len(new_ids),
        "total_games": len(out),
        "ratings_path": str(RATINGS_JSON),
        "features_path": str(FEATURES_CSV),
        "ok": True,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    full = "--full" in sys.argv
    refresh(full=full)
