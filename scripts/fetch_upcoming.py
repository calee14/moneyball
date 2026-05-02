"""
fetch_upcoming.py — Pulls today's and tomorrow's MLB schedule from the
Stats API, including probable pitchers and game status.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass, field
from datetime import date, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

STATUS_LABEL = {
    "Preview": "Upcoming",
    "Live": "Live",
    "Final": "Final",
}


@dataclass
class UpcomingGame:
    game_pk: int
    date: str          # "2026-04-28"
    game_time_et: str  # "7:05 PM ET"
    away_team: str
    home_team: str
    away_sp: str       # "TBD" if not announced
    home_sp: str
    status: str        # "Upcoming" | "Live" | "Final"
    away_score: int | None = None
    home_score: int | None = None
    away_sp_id: int | None = None  # MLB pitcher ID, None if TBD
    home_sp_id: int | None = None


def fetch_upcoming_games(
    days: int = 2,
    session: requests.Session | None = None,
) -> list[UpcomingGame]:
    """
    Return MLB games for today and the next `days-1` days.
    Games are sorted: today first, then tomorrow; within each day by game time.
    """
    sess = session or requests.Session()
    today = date.today()
    end = today + timedelta(days=days - 1)

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1"
        f"&startDate={today.isoformat()}"
        f"&endDate={end.isoformat()}"
        f"&gameType=R"
        f"&hydrate=probablePitcher"
    )

    resp = sess.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games: list[UpcomingGame] = []
    for date_obj in data.get("dates", []):
        for g in date_obj["games"]:
            teams = g["teams"]
            away = teams["away"]
            home = teams["home"]

            abstract_state = g["status"]["abstractGameState"]
            status = STATUS_LABEL.get(abstract_state, abstract_state)

            # Parse game time into ET
            raw_time = g.get("gameDate", "")
            try:
                from datetime import datetime
                dt_utc = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                dt_et = dt_utc.astimezone(ET)
                game_time_et = dt_et.strftime("%-I:%M %p ET")
            except Exception:
                game_time_et = "TBD"

            away_score = away.get("score")
            home_score = home.get("score")

            away_pp = away.get("probablePitcher", {})
            home_pp = home.get("probablePitcher", {})
            away_sp_id_raw = away_pp.get("id")
            home_sp_id_raw = home_pp.get("id")

            games.append(
                UpcomingGame(
                    game_pk=g["gamePk"],
                    date=date_obj["date"],
                    game_time_et=game_time_et,
                    away_team=away["team"]["name"],
                    home_team=home["team"]["name"],
                    away_sp=away_pp.get("fullName", "TBD"),
                    home_sp=home_pp.get("fullName", "TBD"),
                    status=status,
                    away_score=away_score if status != "Upcoming" else None,
                    home_score=home_score if status != "Upcoming" else None,
                    away_sp_id=int(away_sp_id_raw) if away_sp_id_raw is not None else None,
                    home_sp_id=int(home_sp_id_raw) if home_sp_id_raw is not None else None,
                )
            )

    return games


if __name__ == "__main__":
    for game in fetch_upcoming_games():
        score = ""
        if game.away_score is not None:
            score = f"  {game.away_score}-{game.home_score}"
        print(
            f"[{game.date}] {game.game_time_et:>12}  "
            f"{game.away_team:>25} @ {game.home_team:<25}  "
            f"{game.away_sp:<22} vs {game.home_sp:<22}  [{game.status}]{score}"
        )
