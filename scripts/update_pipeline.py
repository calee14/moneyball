"""
update_pipeline.py — Incremental data refresh + preprocessing.

Steps:
  1. Read the existing historical CSV to find the last recorded game date.
  2. Fetch any completed games from that date onward using the MLB Stats API
     (same logic as collect_data.py, but scoped to the missing window).
  3. Deduplicate by Game_ID and append only new rows.
  4. Re-run preprocessing to regenerate mlb_model_ready.csv.

Can be run standalone:
    python scripts/update_pipeline.py

Or imported and called from the TUI:
    from update_pipeline import run_update
    new_games = run_update(progress_cb=lambda msg: ...)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
RAW_CSV   = ROOT / "data" / "mlb_historical_games.csv"
MODEL_CSV = ROOT / "data" / "mlb_model_ready.csv"

# Re-use the existing collect/preprocess logic
sys.path.insert(0, str(Path(__file__).parent))
from collect_data import fetch_game
from preprocess import build_predictive_dataset

RAW_HEADER = [
    "Game_ID", "Date",
    "Away_Team", "Home_Team", "Away_SP", "Home_SP",
    "Away_Score", "Home_Score",
    "Away_AB", "Away_Hits", "Away_2B", "Away_3B", "Away_HR",
    "Away_BB", "Away_HBP", "Away_SF", "Away_K",
    "Home_AB", "Home_Hits", "Home_2B", "Home_3B", "Home_HR",
    "Home_BB", "Home_HBP", "Home_SF", "Home_K",
    "Away_Team_ER", "Away_Team_IP",
    "Home_Team_ER", "Home_Team_IP",
    "Away_SP_ER", "Away_SP_IP", "Away_SP_K", "Away_SP_BB", "Away_SP_HR",
    "Home_SP_ER", "Home_SP_IP", "Home_SP_K", "Home_SP_BB", "Home_SP_HR",
    "Home_Win",
]


# ── Schedule fetch ────────────────────────────────────────────────────────────

def _get_completed_game_ids(
    start: date,
    end: date,
    session: requests.Session,
) -> list[int]:
    """Return game PKs for all completed regular-season games in [start, end]."""
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1"
        f"&startDate={start.isoformat()}"
        f"&endDate={end.isoformat()}"
        f"&gameType=R"
    )
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    ids: list[int] = []
    for date_obj in data.get("dates", []):
        for g in date_obj["games"]:
            if g["status"]["statusCode"] in ("F", "O"):   # Final / Official
                ids.append(g["gamePk"])
    return ids


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_update(
    progress_cb: Callable[[str], None] | None = None,
    max_workers: int = 10,
) -> int:
    """
    Fetch any missing completed games, append to the raw CSV, then rerun
    preprocessing.

    Returns the number of new games added.
    """
    log = progress_cb or print

    # ── 1. Determine the window to fetch ─────────────────────────────────────
    if not RAW_CSV.exists():
        log("No existing data found — please run collect_data.py for a full pull.")
        return 0

    existing = pd.read_csv(RAW_CSV, usecols=["Game_ID", "Date"])
    existing["Date"] = pd.to_datetime(existing["Date"])
    existing_ids: set[int] = set(existing["Game_ID"].tolist())

    last_date: date = existing["Date"].max().date()
    # Start from the last recorded date in case some games on that day were
    # missed (e.g. double-headers added late).
    fetch_start = last_date
    fetch_end   = date.today()        # today — we only want *completed* games

    if fetch_start > fetch_end:
        log("Data is already up to date.")
        return 0

    log(f"Checking for new games from {fetch_start} → {fetch_end}…")

    # ── 2. Fetch schedule for the window ─────────────────────────────────────
    session = requests.Session()
    candidate_ids = _get_completed_game_ids(fetch_start, fetch_end, session)
    new_ids = [pk for pk in candidate_ids if pk not in existing_ids]

    if not new_ids:
        log("No new completed games found.")
        return 0

    log(f"Found {len(new_ids)} new game(s) to fetch…")

    # ── 3. Fetch box scores in parallel ──────────────────────────────────────
    rows: list[list] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_game, pk): pk for pk in new_ids}
        for future in as_completed(futures):
            done += 1
            row = future.result()
            if row is not None:
                rows.append(row)
            if done % 10 == 0 or done == len(new_ids):
                log(f"  Fetched {done}/{len(new_ids)} games…")

    if not rows:
        log("All fetches failed or returned no data.")
        return 0

    # ── 4. Append to raw CSV ──────────────────────────────────────────────────
    new_df = pd.DataFrame(rows, columns=RAW_HEADER)
    new_df["Date"] = pd.to_datetime(new_df["Date"])

    # Final dedup guard
    new_df = new_df[~new_df["Game_ID"].isin(existing_ids)]
    added = len(new_df)

    if added == 0:
        log("No new rows after deduplication.")
        return 0

    # Load full raw CSV, concat, sort, save
    full_df = pd.read_csv(RAW_CSV)
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    combined = (
        pd.concat([full_df, new_df], ignore_index=True)
        .sort_values(["Date", "Game_ID"])
        .reset_index(drop=True)
    )
    combined.to_csv(RAW_CSV, index=False)
    log(f"Saved {added} new game(s) to {RAW_CSV.name}  (total: {len(combined)})")

    # ── 5. Re-run preprocessing ───────────────────────────────────────────────
    log("Re-running preprocessing…")
    model_df = build_predictive_dataset(str(RAW_CSV))
    model_df.to_csv(MODEL_CSV, index=False)
    log(f"Model-ready data saved to {MODEL_CSV.name}  ({len(model_df)} rows)")

    return added


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    added = run_update()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  —  {added} new game(s) added.")
