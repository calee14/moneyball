"""
elo_predictor.py — Pure team Elo prediction + team context for the TUI.

Loads:
  - data/ratings_snapshot.json  (saved Elo ratings from elo_engine)
  - data/mlb_games_with_features.csv  (collected + feature-engineered games)

For any (away_team, home_team) pair, returns:
  - Elo win probability (pure team Elo only — no starter, no travel)
  - Per-team context: last 10 record, run diff, bullpen IP, rest,
    travel, starter recent Game Score 2.0

The win probability uses ONLY team Elo + home field advantage, per spec.
Travel/rest/bullpen are surfaced in the side panel for gut-check, not
folded into the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd


# ── constants (must match elo_engine.py) ──────────────────────────────────────
INITIAL_ELO = 1500
HOME_FIELD_ADV = 24  # ~54% home win rate baseline


def _safe_int(x) -> Optional[int]:
    """Coerce to int, accepting float / float-string / int / int-string. None on failure."""
    if x is None:
        return None
    try:
        # float() handles '657277.0', 657277.0, 657277, '657277'
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _parse_ip(s) -> float:
    """Convert MLB innings-pitched string/number to decimal (e.g. '6.2' -> 6.667)."""
    try:
        ip = str(s)
        if "." in ip:
            w, fr = ip.split(".", 1)
            return int(w) + int(fr) / 3.0
        return float(ip)
    except Exception:
        return 0.0


# ── data structures ───────────────────────────────────────────────────────────
@dataclass
class TeamContext:
    """All side-panel info for a single team."""

    team_id: int
    team_name: str
    elo: float
    rank: int  # 1 = best in league
    last_10_wins: int
    last_10_losses: int
    last_10_run_diff: int  # cumulative runs scored - allowed, last 10
    home_record: tuple[int, int]  # (W, L) at home this season
    away_record: tuple[int, int]
    days_rest: float  # days since last game
    travel_miles: float  # miles travelled to today's venue
    bullpen_ip_last3: float  # bullpen innings last 3 games
    is_day_after_night: bool
    starter_recent_gs2: Optional[float]  # avg GS2 last 3 starts, None if unknown
    starter_starts_seen: int
    starter_fip: Optional[float]  # rolling pre-game FIP, None if unknown
    starter_k9: Optional[float]   # rolling pre-game K/9, None if unknown
    starter_era_recent: Optional[float]  # ER/IP * 9 over last 3 starts, None if unknown
    starter_era_n: int                   # number of starts used for era_recent
    matchup_rpg: Optional[float]  # this team's runs/game vs opp SP historically
    matchup_delta: Optional[float]  # vs this team's season runs/game average
    matchup_n: int                  # number of historical matchup games


@dataclass
class GamePrediction:
    """Result returned to the TUI."""

    away: TeamContext
    home: TeamContext
    p_home_win: float  # pure Elo + home field
    p_away_win: float
    elo_diff: float  # home_elo - away_elo (signed)
    pick_team: str  # "home" or "away" — favored side
    pick_confidence: str  # "Lean", "Moderate", "Strong"
    notes: list[str] = field(default_factory=list)  # gut-check flags


# ── pure team Elo win probability ─────────────────────────────────────────────
def win_prob_pure(home_elo: float, away_elo: float) -> float:
    """P(home wins) using only team Elo and home field advantage."""
    diff = (home_elo - away_elo) + HOME_FIELD_ADV
    return 1.0 / (1.0 + 10 ** (-diff / 400))


def confidence_label(p: float) -> str:
    """Convert win probability to a human label."""
    edge = abs(p - 0.5)
    if edge < 0.04:
        return "Toss-up"
    if edge < 0.08:
        return "Lean"
    if edge < 0.13:
        return "Moderate"
    return "Strong"


# ── predictor ─────────────────────────────────────────────────────────────────
class EloPredictor:
    """Loads snapshot + CSV once, then answers prediction queries quickly."""

    def __init__(
        self,
        snapshot_path: str = "data/ratings_snapshot.json",
        features_csv: str = "data/mlb_games_with_features.csv",
        model_ready_csv: str = "data/mlb_model_ready.csv",
    ):
        self.snapshot_path = Path(snapshot_path)
        self.features_csv = Path(features_csv)
        self.model_ready_csv = Path(model_ready_csv)

        if not self.snapshot_path.exists():
            raise FileNotFoundError(
                f"Ratings snapshot not found: {self.snapshot_path}\n"
                f"Run: python elo_engine.py {self.features_csv}"
            )
        if not self.features_csv.exists():
            raise FileNotFoundError(
                f"Features CSV not found: {self.features_csv}\n"
                f"Run collect_data.py and travel_rest.py first."
            )

        with open(self.snapshot_path) as f:
            snap = json.load(f)
        # JSON keys are strings — convert team IDs back to int (handles float-strings too)
        self.team_elo: dict[int, float] = {
            _safe_int(k): float(v)
            for k, v in snap["team_elo"].items()
            if _safe_int(k) is not None
        }
        self.sp_recent_gs2: dict[int, list[float]] = {
            _safe_int(k): list(v)
            for k, v in snap.get("sp_recent_gs2", {}).items()
            if _safe_int(k) is not None
        }

        self.df = pd.read_csv(self.features_csv)
        self.df["_dt"] = pd.to_datetime(
            self.df["Game_DateTime"], errors="coerce", utc=True
        )
        self.df = self.df.sort_values("_dt").reset_index(drop=True)

        # Load model-ready CSV for per-pitcher FIP/K9 lookup (by name)
        # Build a dict: pitcher_name -> {"fip": float, "k9": float}
        self._sp_fip: dict[str, tuple[float, float]] = {}
        if self.model_ready_csv.exists():
            mr = pd.read_csv(self.model_ready_csv)
            if "Date" not in mr.columns:
                mr = mr.sort_index()
            else:
                mr = mr.sort_values("Date")
            for side, name_col, fip_col, k9_col in (
                ("Home", "Home_SP", "Home_SP_PreGame_FIP", "Home_SP_PreGame_K9"),
                ("Away", "Away_SP", "Away_SP_PreGame_FIP", "Away_SP_PreGame_K9"),
            ):
                if name_col not in mr.columns:
                    continue
                subset = mr[[name_col, fip_col, k9_col]].dropna()
                for _, row in subset.iterrows():
                    name = row[name_col]
                    fip = float(row[fip_col])
                    k9 = float(row[k9_col])
                    # Zero FIP/K9 means no prior data (first start of season) — skip
                    if fip != 0.0 or k9 != 0.0:
                        self._sp_fip[name] = (fip, k9)

        # Build name -> team_id lookup (covers both away and home appearances)
        name_to_id: dict[str, int] = {}
        for _, row in (
            self.df[["Away_Team", "Away_Team_ID"]].drop_duplicates().iterrows()
        ):
            tid = _safe_int(row["Away_Team_ID"])
            if tid is not None:
                name_to_id[row["Away_Team"]] = tid
        for _, row in (
            self.df[["Home_Team", "Home_Team_ID"]].drop_duplicates().iterrows()
        ):
            tid = _safe_int(row["Home_Team_ID"])
            if tid is not None:
                name_to_id[row["Home_Team"]] = tid
        self.name_to_id = name_to_id

        # Pre-compute league rank
        self._rank_by_id = {
            tid: rank
            for rank, (tid, _elo) in enumerate(
                sorted(self.team_elo.items(), key=lambda x: -x[1]), start=1
            )
        }

        # Current season for filtering "this season" stats
        self.current_season = (
            self.df["_dt"].max().year if len(self.df) else date.today().year
        )

        # Build team-vs-SP matchup table from the historical games CSV.
        # Key: (batting_team_id: int, opp_sp_id: int) -> list[float] of runs scored.
        # Also keep a by-name version for when we only have SP name.
        # Structure: (batting_team_id, opp_sp_name) -> [runs, ...]
        self._matchup_by_id: dict[tuple[int, int], list[float]] = {}
        self._matchup_by_name: dict[tuple[int, str], list[float]] = {}
        # Per-team season run average for computing deltas
        self._team_rpg: dict[int, float] = {}

        raw = pd.read_csv(self.snapshot_path.parent.parent / "data" / "mlb_historical_games.csv")
        cur_yr = self.current_season
        season_raw = raw[pd.to_datetime(raw["Date"], errors="coerce").dt.year == cur_yr]

        # Team season runs/game average (current season)
        for tid in self.name_to_id.values():
            away_runs = season_raw[season_raw["Away_Team_ID"] == tid]["Away_Score"].dropna()
            home_runs = season_raw[season_raw["Home_Team_ID"] == tid]["Home_Score"].dropna()
            all_runs = pd.concat([away_runs, home_runs])
            if len(all_runs) >= 5:
                self._team_rpg[tid] = float(all_runs.mean())

        # Build SP recent ER/IP lookup from raw game logs.
        # Stores (date, er, ip_decimal) tuples per pitcher, sorted chronologically.
        # Key: sp_id (int) or sp_name (str)
        self._sp_er_by_id: dict[int, list[tuple[float, float]]] = {}   # id -> [(er, ip), ...]
        self._sp_er_by_name: dict[str, list[tuple[float, float]]] = {}  # name -> [(er, ip), ...]

        sp_er_cols = {"Away_SP_ID", "Away_SP", "Away_SP_ER", "Away_SP_IP",
                      "Home_SP_ID", "Home_SP", "Home_SP_ER", "Home_SP_IP", "Date"}
        if sp_er_cols.issubset(set(raw.columns)):
            raw_sorted = raw.sort_values("Date", na_position="last")
            for _, r in raw_sorted.iterrows():
                for side in ("Away", "Home"):
                    sp_id = _safe_int(r[f"{side}_SP_ID"])
                    sp_name = str(r[f"{side}_SP"]) if pd.notna(r[f"{side}_SP"]) else ""
                    er_raw = r[f"{side}_SP_ER"]
                    ip_raw = r[f"{side}_SP_IP"]
                    if not pd.notna(er_raw) or not pd.notna(ip_raw):
                        continue
                    er = float(er_raw)
                    ip = _parse_ip(ip_raw)
                    if ip <= 0:
                        continue
                    entry = (er, ip)
                    if sp_id is not None:
                        self._sp_er_by_id.setdefault(sp_id, []).append(entry)
                    if sp_name:
                        self._sp_er_by_name.setdefault(sp_name, []).append(entry)

        # Build matchup lookup: away team batted vs home SP; home team batted vs away SP
        needed_cols = {"Away_Team_ID", "Away_Score", "Home_SP_ID", "Home_SP",
                       "Home_Team_ID", "Home_Score", "Away_SP_ID", "Away_SP"}
        if needed_cols.issubset(set(raw.columns)):
            for _, r in raw.iterrows():
                # Away team batted against Home SP
                a_tid = _safe_int(r["Away_Team_ID"])
                h_sp_id = _safe_int(r["Home_SP_ID"])
                h_sp_name = str(r["Home_SP"]) if pd.notna(r["Home_SP"]) else ""
                a_runs = r["Away_Score"]
                if a_tid is not None and pd.notna(a_runs):
                    a_runs_f = float(a_runs)
                    if h_sp_id is not None:
                        self._matchup_by_id.setdefault((a_tid, h_sp_id), []).append(a_runs_f)
                    if h_sp_name:
                        self._matchup_by_name.setdefault((a_tid, h_sp_name), []).append(a_runs_f)

                # Home team batted against Away SP
                h_tid = _safe_int(r["Home_Team_ID"])
                a_sp_id = _safe_int(r["Away_SP_ID"])
                a_sp_name = str(r["Away_SP"]) if pd.notna(r["Away_SP"]) else ""
                h_runs = r["Home_Score"]
                if h_tid is not None and pd.notna(h_runs):
                    h_runs_f = float(h_runs)
                    if a_sp_id is not None:
                        self._matchup_by_id.setdefault((h_tid, a_sp_id), []).append(h_runs_f)
                    if a_sp_name:
                        self._matchup_by_name.setdefault((h_tid, a_sp_name), []).append(h_runs_f)

    # ── lookups ───────────────────────────────────────────────────────────────
    def resolve_team(self, name: str) -> Optional[int]:
        """Map a team name (or partial) to MLB team_id. Case-insensitive."""
        if name in self.name_to_id:
            return self.name_to_id[name]
        lower = name.lower()
        for full_name, tid in self.name_to_id.items():
            if lower in full_name.lower() or full_name.lower() in lower:
                return tid
        return None

    # ── per-team context ──────────────────────────────────────────────────────
    def _team_history(self, team_id: int) -> pd.DataFrame:
        """All games where this team played, sorted chronologically."""
        m = (self.df["Away_Team_ID"] == team_id) | (self.df["Home_Team_ID"] == team_id)
        return self.df[m].sort_values("_dt").reset_index(drop=True)

    def _team_perspective(self, hist: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """Add team-perspective columns: did_win, runs_for, runs_against, is_home."""
        hist = hist.copy()
        hist["is_home"] = hist["Home_Team_ID"] == team_id
        hist["did_win"] = (
            ((hist["is_home"]) & (hist["Home_Win"] == 1))
            | ((~hist["is_home"]) & (hist["Home_Win"] == 0))
        ).astype(int)
        hist["runs_for"] = hist.apply(
            lambda r: r["Home_Score"] if r["is_home"] else r["Away_Score"], axis=1
        )
        hist["runs_against"] = hist.apply(
            lambda r: r["Away_Score"] if r["is_home"] else r["Home_Score"], axis=1
        )
        return hist

    def build_context(self, team_id: int, team_name: str) -> TeamContext:
        hist = self._team_history(team_id)
        if len(hist) == 0:
            # Team has no games — return defaults
            return TeamContext(
                team_id=team_id,
                team_name=team_name,
                elo=self.team_elo.get(team_id, INITIAL_ELO),
                rank=self._rank_by_id.get(team_id, 0),
                last_10_wins=0,
                last_10_losses=0,
                last_10_run_diff=0,
                home_record=(0, 0),
                away_record=(0, 0),
                days_rest=float("nan"),
                travel_miles=0.0,
                bullpen_ip_last3=float("nan"),
                is_day_after_night=False,
                starter_recent_gs2=None,
                starter_starts_seen=0,
                starter_fip=None,
                starter_k9=None,
                starter_era_recent=None,
                starter_era_n=0,
                matchup_rpg=None,
                matchup_delta=None,
                matchup_n=0,
            )

        hist = self._team_perspective(hist, team_id)

        # Last 10 (across all seasons in CSV — most recent 10)
        last10 = hist.tail(10)
        l10_w = int(last10["did_win"].sum())
        l10_l = len(last10) - l10_w
        l10_diff = int(last10["runs_for"].sum() - last10["runs_against"].sum())

        # Season home/away records (current season only)
        season = hist[hist["_dt"].dt.year == self.current_season]
        home_g = season[season["is_home"]]
        away_g = season[~season["is_home"]]
        home_rec = (
            int(home_g["did_win"].sum()),
            len(home_g) - int(home_g["did_win"].sum()),
        )
        away_rec = (
            int(away_g["did_win"].sum()),
            len(away_g) - int(away_g["did_win"].sum()),
        )

        # Most recent game determines rest/travel/bullpen for next game.
        # The "next game" features are pre-computed per game in the features CSV
        # (Away_days_rest, Home_days_rest etc.) but those are tied to a specific
        # past game. For a hypothetical upcoming game, we use the last observed
        # bullpen IP & day-of-game as the team's *current* state.
        last_game = hist.iloc[-1]
        last_dt = last_game["_dt"]
        # Days since last game (relative to today)
        days_rest = (pd.Timestamp.now(tz="UTC") - last_dt).total_seconds() / 86400
        # Bullpen last 3: read from the per-game feature col for the last game's
        # post-game state. Easier path: recompute from team_ip - sp_ip over the
        # last 3 games.
        last3 = hist.tail(3)

        def _ip(s):
            try:
                ip = str(s)
                if "." in ip:
                    w, fr = ip.split(".")
                    return int(w) + int(fr) / 3.0
                return float(ip)
            except Exception:
                return 0.0

        bullpen_ip_last3 = 0.0
        for _, r in last3.iterrows():
            if r["is_home"]:
                bullpen_ip_last3 += _ip(r["Home_Team_IP"]) - _ip(r["Home_SP_IP"])
            else:
                bullpen_ip_last3 += _ip(r["Away_Team_IP"]) - _ip(r["Away_SP_IP"])

        # Day-after-night flag for next game vs last game
        last_dn = last_game.get("Day_Night", "unknown")
        # Without knowing the upcoming game's day/night, we just flag if the
        # last one was a night game and they have <1.5 days rest
        is_dan = (last_dn == "night") and (days_rest < 1.5)

        # Travel miles: needs the upcoming game's venue. The TUI passes only
        # team names, so we leave this as 0 for the hypothetical case.
        # (Could be enhanced to take venue lat/lon as args.)
        travel_miles = 0.0

        # Starter recent form not relevant for team-level context (handled
        # separately via starter_for() if requested).

        return TeamContext(
            team_id=team_id,
            team_name=team_name,
            elo=self.team_elo.get(team_id, INITIAL_ELO),
            rank=self._rank_by_id.get(team_id, 0),
            last_10_wins=l10_w,
            last_10_losses=l10_l,
            last_10_run_diff=l10_diff,
            home_record=home_rec,
            away_record=away_rec,
            days_rest=round(days_rest, 1),
            travel_miles=travel_miles,
            bullpen_ip_last3=round(bullpen_ip_last3, 1),
            is_day_after_night=is_dan,
            starter_recent_gs2=None,  # filled in by predict() if SP id given
            starter_starts_seen=0,
            starter_fip=None,         # filled in by predict() if SP name given
            starter_k9=None,
            starter_era_recent=None,  # filled in by predict() if SP name given
            starter_era_n=0,
            matchup_rpg=None,         # filled in by predict() via team_vs_sp()
            matchup_delta=None,
            matchup_n=0,
        )

    def starter_form(self, sp_id) -> tuple[Optional[float], int]:
        """Return (avg recent GS2, n starts seen) for a starter."""
        sp_id_int = _safe_int(sp_id)
        if sp_id_int is None:
            return (None, 0)
        history = self.sp_recent_gs2.get(sp_id_int, [])
        if not history:
            return (None, 0)
        recent = history[-3:]
        return (round(sum(recent) / len(recent), 1), len(history))

    def starter_fip_by_name(self, sp_name: str) -> tuple[Optional[float], Optional[float]]:
        """Return (pre-game FIP, pre-game K/9) for a starter looked up by name."""
        if not sp_name or sp_name == "TBD":
            return (None, None)
        entry = self._sp_fip.get(sp_name)
        if entry is None:
            return (None, None)
        fip, k9 = entry
        return (round(fip, 2), round(k9, 1))

    def starter_era_recent_by_id_or_name(
        self,
        sp_id=None,
        sp_name: str = "",
        n_starts: int = 3,
    ) -> tuple[Optional[float], int]:
        """
        Return (recent ERA, n_starts_used) for a starter from their last `n_starts`
        outings.  ERA = (total ER / total IP) * 9.  Prefers ID lookup, falls back
        to name.  Returns (None, 0) when the pitcher has no data.
        """
        starts: list[tuple[float, float]] = []

        sp_id_int = _safe_int(sp_id)
        if sp_id_int is not None:
            starts = self._sp_er_by_id.get(sp_id_int, [])

        if not starts and sp_name and sp_name not in ("TBD", ""):
            starts = self._sp_er_by_name.get(sp_name, [])

        if not starts:
            return (None, 0)

        recent = starts[-n_starts:]
        total_er = sum(er for er, _ip in recent)
        total_ip = sum(ip for _er, ip in recent)
        if total_ip == 0:
            return (None, 0)

        era = round((total_er / total_ip) * 9, 2)
        return (era, len(recent))

    def team_vs_sp(
        self,
        batting_team_id: int,
        opp_sp_id: Optional[int],
        opp_sp_name: str,
        min_games: int = 2,
    ) -> tuple[Optional[float], Optional[float], int]:
        """
        Return (runs_per_game, delta_vs_season_avg, n_games) for a batting team
        against a specific opposing starting pitcher.

        Prefers ID-based lookup; falls back to name if ID not found.
        Returns (None, None, 0) when fewer than min_games exist.
        """
        runs: list[float] = []

        # Try ID lookup first
        if opp_sp_id is not None:
            runs = self._matchup_by_id.get((batting_team_id, opp_sp_id), [])

        # Fall back to name lookup
        if not runs and opp_sp_name and opp_sp_name != "TBD":
            runs = self._matchup_by_name.get((batting_team_id, opp_sp_name), [])

        if len(runs) < min_games:
            return (None, None, len(runs))

        rpg = round(sum(runs) / len(runs), 2)
        season_avg = self._team_rpg.get(batting_team_id)
        delta = round(rpg - season_avg, 2) if season_avg is not None else None
        return (rpg, delta, len(runs))

    # ── main entry point ──────────────────────────────────────────────────────
    def predict(
        self,
        away_team: str,
        home_team: str,
        away_sp_id=None,
        home_sp_id=None,
        away_sp: str = "",
        home_sp: str = "",
    ) -> GamePrediction:
        away_id = self.resolve_team(away_team)
        home_id = self.resolve_team(home_team)
        if away_id is None:
            raise ValueError(f"Unknown team: {away_team!r}")
        if home_id is None:
            raise ValueError(f"Unknown team: {home_team!r}")

        away_ctx = self.build_context(away_id, away_team)
        home_ctx = self.build_context(home_id, home_team)

        # Starter recent form (informational only — NOT in win prob)
        if away_sp_id is not None:
            gs, n = self.starter_form(away_sp_id)
            away_ctx.starter_recent_gs2 = gs
            away_ctx.starter_starts_seen = n
        if home_sp_id is not None:
            gs, n = self.starter_form(home_sp_id)
            home_ctx.starter_recent_gs2 = gs
            home_ctx.starter_starts_seen = n

        # FIP/K9 fallback — looked up by name, available whenever pitcher is announced
        away_fip, away_k9 = self.starter_fip_by_name(away_sp)
        home_fip, home_k9 = self.starter_fip_by_name(home_sp)
        away_ctx.starter_fip = away_fip
        away_ctx.starter_k9 = away_k9
        home_ctx.starter_fip = home_fip
        home_ctx.starter_k9 = home_k9

        # Recent ERA (ER/IP * 9) over last 3 starts — primary SP stat shown in TUI
        away_era, away_era_n = self.starter_era_recent_by_id_or_name(away_sp_id, away_sp)
        home_era, home_era_n = self.starter_era_recent_by_id_or_name(home_sp_id, home_sp)
        away_ctx.starter_era_recent = away_era
        away_ctx.starter_era_n = away_era_n
        home_ctx.starter_era_recent = home_era
        home_ctx.starter_era_n = home_era_n

        # Matchup history: away team bats against home SP; home team bats against away SP
        away_rpg, away_delta, away_n = self.team_vs_sp(away_id, home_sp_id, home_sp)
        home_rpg, home_delta, home_n = self.team_vs_sp(home_id, away_sp_id, away_sp)
        away_ctx.matchup_rpg = away_rpg
        away_ctx.matchup_delta = away_delta
        away_ctx.matchup_n = away_n
        home_ctx.matchup_rpg = home_rpg
        home_ctx.matchup_delta = home_delta
        home_ctx.matchup_n = home_n

        # Pure team Elo win probability
        p_home = win_prob_pure(home_ctx.elo, away_ctx.elo)
        elo_diff = home_ctx.elo - away_ctx.elo

        pick_team = "home" if p_home >= 0.5 else "away"
        confidence = confidence_label(p_home)

        notes = self._gut_check_notes(away_ctx, home_ctx)

        return GamePrediction(
            away=away_ctx,
            home=home_ctx,
            p_home_win=p_home,
            p_away_win=1 - p_home,
            elo_diff=elo_diff,
            pick_team=pick_team,
            pick_confidence=confidence,
            notes=notes,
        )

    # ── gut-check note generator ──────────────────────────────────────────────
    @staticmethod
    def _gut_check_notes(away: TeamContext, home: TeamContext) -> list[str]:
        """Surface flags worth noticing that aren't in the Elo number."""
        notes = []
        # Stale data warning — if both teams' "last game" was a long time ago,
        # the CSV hasn't been refreshed and side-panel info is unreliable.
        if (
            not _isnan_float(away.days_rest)
            and not _isnan_float(home.days_rest)
            and min(away.days_rest, home.days_rest) > 7
        ):
            notes.append(
                f"⚠ Data may be stale (last game in CSV is "
                f"{min(away.days_rest, home.days_rest):.0f} days old)"
            )
            return notes  # Don't bother with the rest of the gut-check noise

        # Bullpen depletion
        if home.bullpen_ip_last3 > 12:
            notes.append(
                f"⚠ {home.team_name} bullpen heavily used "
                f"({home.bullpen_ip_last3} IP last 3)"
            )
        if away.bullpen_ip_last3 > 12:
            notes.append(
                f"⚠ {away.team_name} bullpen heavily used "
                f"({away.bullpen_ip_last3} IP last 3)"
            )
        # Rest extremes (3-7 day window — past 7 days means stale data, not rested)
        if 3 < home.days_rest <= 7:
            notes.append(f"✓ {home.team_name} well rested ({home.days_rest:.1f} days)")
        if 3 < away.days_rest <= 7:
            notes.append(f"✓ {away.team_name} well rested ({away.days_rest:.1f} days)")
        # Hot/cold streaks (last 10)
        if home.last_10_wins >= 8:
            notes.append(
                f"🔥 {home.team_name} hot ({home.last_10_wins}-{home.last_10_losses} L10)"
            )
        if away.last_10_wins >= 8:
            notes.append(
                f"🔥 {away.team_name} hot ({away.last_10_wins}-{away.last_10_losses} L10)"
            )
        if home.last_10_losses >= 8:
            notes.append(
                f"❄ {home.team_name} cold ({home.last_10_wins}-{home.last_10_losses} L10)"
            )
        if away.last_10_losses >= 8:
            notes.append(
                f"❄ {away.team_name} cold ({away.last_10_wins}-{away.last_10_losses} L10)"
            )
        # Run diff signals real strength vs lucky wins
        if abs(home.last_10_run_diff) > 25:
            sign = "+" if home.last_10_run_diff > 0 else ""
            notes.append(
                f"  {home.team_name} L10 run diff: {sign}{home.last_10_run_diff}"
            )
        if abs(away.last_10_run_diff) > 25:
            sign = "+" if away.last_10_run_diff > 0 else ""
            notes.append(
                f"  {away.team_name} L10 run diff: {sign}{away.last_10_run_diff}"
            )
        return notes


def _isnan_float(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


# ── module-level convenience ──────────────────────────────────────────────────
_predictor: Optional[EloPredictor] = None


def get_predictor() -> EloPredictor:
    """Lazy singleton — loads the snapshot once per process."""
    global _predictor
    if _predictor is None:
        _predictor = EloPredictor()
    return _predictor
