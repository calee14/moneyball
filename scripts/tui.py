"""
tui.py — Moneyball TUI

On launch:
  1. Fetches any missing completed games and re-runs preprocessing (background).
  2. Loads today's + tomorrow's schedule with probable pitchers.
  3. Browse games, press Enter/R to run the model on any matchup.

Controls:
  ↑ / ↓       — navigate games
  Enter / R   — run inference on selected game
  T           — cycle filter (All / Today / Tomorrow)
  Q / Ctrl+C  — quit
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Log, Static
from textual.worker import Worker, WorkerState

from fetch_upcoming import UpcomingGame, fetch_upcoming_games
from inference import build_feature_vector

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "xgb_model.joblib"
META_PATH  = ROOT / "models" / "model_meta.json"
DATA_PATH  = ROOT / "data" / "mlb_model_ready.csv"

# ── misc constants ────────────────────────────────────────────────────────────
TODAY    = date.today().isoformat()
TOMORROW = (date.today() + timedelta(days=1)).isoformat()

STATUS_STYLE = {
    "Upcoming": "bold cyan",
    "Live":     "bold yellow",
    "Final":    "dim",
}
CONF_STYLE = {
    "high":   "bold green",
    "medium": "bold yellow",
    "low":    "dim white",
}
FEAT_LABELS = {
    # Park context
    "Park_Factor":          "Park Factor",
    # Long-term differentials
    "OPS_Diff":             "OPS Diff L15 (Home − Away)",
    "K_Rate_Diff":          "K-Rate Diff L15 (Home − Away)",
    "Bullpen_ERA_Diff":     "Bullpen ERA Diff (Away − Home)",
    "SP_FIP_Diff":          "SP FIP Diff (Away − Home)",
    "SP_K9_Diff":           "SP K/9 Diff L15 (Home − Away)",
    "Bullpen_Fatigue_Diff": "Bullpen Fatigue (Away − Home)",
    # Short-term / streak differentials
    "OPS_Diff_S":           "OPS Diff L5 streak (Home − Away)",
    "K_Rate_Diff_S":        "K-Rate Diff L5 streak (Home − Away)",
    "SP_K9_Diff_S":         "SP K/9 Diff recent (Home − Away)",
    # Run differential
    "RunDiff_Diff":         "Run Diff EWMA (Home − Away)",
    # Rest
    "Rest_Diff":            "Rest Days Diff (Home − Away)",
    "SP_Rest_Diff":         "SP Rest Days Diff (Home − Away)",
    # Home field signal
    "Home_Win_Rate":        "Home Win Rate L20 (home team)",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _conf_tier(p: float) -> str:
    if p >= 0.60: return "high"
    if p >= 0.53: return "medium"
    return "low"

def _bar(p: float, w: int = 20) -> str:
    n = round(p * w)
    return "█" * n + "░" * (w - n)


# ── blocking inference (called inside a thread worker) ────────────────────────
def _run_inference(game: UpcomingGame) -> dict:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    X = build_feature_vector(
        away_team=game.away_team,
        home_team=game.home_team,
        away_sp=game.away_sp,
        home_sp=game.home_sp,
        df=df,
        meta=meta,
    )
    hp = float(model.predict_proba(X)[0][1])
    ap = 1.0 - hp
    feats = {f: float(X.iloc[0][f]) for f in meta["features"]}
    return {"home_prob": hp, "away_prob": ap, "features": feats, "game": game}


# ── blocking update pipeline (called inside a thread worker) ──────────────────
def _run_update(log_q: "queue.Queue[str | None]") -> int:
    """
    Runs the incremental data refresh.  Progress messages are pushed onto
    log_q so the TUI can display them live.  Sends None when done.
    """
    from update_pipeline import run_update
    added = run_update(progress_cb=log_q.put)
    log_q.put(None)   # sentinel
    return added


# ── result panel widget ───────────────────────────────────────────────────────
class ResultPanel(Static):
    DEFAULT_CSS = """
    ResultPanel {
        border: solid $accent;
        padding: 1 2;
        height: 100%;
        width: 1fr;
    }
    """

    def show_placeholder(self) -> None:
        self.update(
            "[dim]Select a game and press [bold]Enter[/bold] or [bold]R[/bold] "
            "to run the model.[/dim]"
        )

    def show_loading(self, game: UpcomingGame) -> None:
        self.update(
            f"[bold]Running model…[/bold]\n\n"
            f"[cyan]{game.away_team}[/cyan]  @  [cyan]{game.home_team}[/cyan]\n"
            f"[dim]{game.away_sp} vs {game.home_sp}[/dim]"
        )

    def show_result(self, result: dict) -> None:
        game: UpcomingGame = result["game"]
        hp, ap = result["home_prob"], result["away_prob"]
        feats  = result["features"]

        ht = _conf_tier(hp); at = _conf_tier(ap)
        fav       = game.home_team if hp >= ap else game.away_team
        fav_prob  = max(hp, ap)
        fav_style = CONF_STYLE[_conf_tier(fav_prob)]

        lines = [
            f"[bold]{game.away_team}[/bold]  @  [bold]{game.home_team}[/bold]",
            f"[dim]{game.away_sp} vs {game.home_sp}  ·  {game.game_time_et}[/dim]",
            "",
            "─" * 44,
            "",
            (f"[bold]{game.away_team:<26}[/bold]"
             f"[{CONF_STYLE[at]}]{ap:>6.1%}[/]  [{CONF_STYLE[at]}]{_bar(ap)}[/]"),
            "",
            (f"[bold]{game.home_team:<26}[/bold]"
             f"[{CONF_STYLE[ht]}]{hp:>6.1%}[/]  [{CONF_STYLE[ht]}]{_bar(hp)}[/]"),
            "",
            "─" * 44,
            "",
            f"  Model favors: [{fav_style}]{fav} ({fav_prob:.1%})[/]",
            "",
            "[dim]── Feature Differentials ──────────────────[/dim]",
        ]
        for key, label in FEAT_LABELS.items():
            v = feats.get(key, 0.0)
            c = "green" if v > 0 else ("red" if v < 0 else "white")
            lines.append(f"  [dim]{label:<30}[/dim] [{c}]{v:+.3f}[/]")

        self.update("\n".join(lines))

    def show_error(self, msg: str) -> None:
        self.update(f"[bold red]Error:[/bold red] {msg}")


# ── main app ──────────────────────────────────────────────────────────────────
class MoneyballApp(App):
    TITLE     = "⚾  Moneyball — MLB Game Predictor"
    SUB_TITLE = "Today & Tomorrow"

    CSS = """
    Screen { layout: vertical; }

    /* ── update log overlay ── */
    #update_panel {
        height: 12;
        border: solid $warning;
        padding: 0 1;
        display: block;
    }
    #update_panel.hidden { display: none; }

    #update_log {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    /* ── toolbar ── */
    #toolbar {
        height: 1;
        background: $boost;
        padding: 0 1;
        color: $text-muted;
    }

    /* ── main split ── */
    #main { layout: horizontal; height: 1fr; }
    #left  { width: 2fr; height: 100%; border: solid $panel; }
    #right { width: 3fr; height: 100%; }

    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q",     "quit",           "Quit"),
        Binding("t",     "toggle_filter",  "Filter"),
        Binding("r",     "run_inference",  "Run Model"),
        Binding("enter", "run_inference",  "Run Model", show=False),
    ]

    # ── state ─────────────────────────────────────────────────────────────────
    _filter:  reactive[str]           = reactive("all")
    _games:   list[UpcomingGame]      = []
    _visible: list[UpcomingGame]      = []
    _update_done: reactive[bool]      = reactive(False)

    def __init__(self, no_update: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._no_update = no_update

    # ── compose ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()

        # update progress panel (visible only during the update worker)
        with Vertical(id="update_panel", classes="hidden" if self._no_update else ""):
            yield Static("[bold yellow]Updating data…[/bold yellow]", id="update_title")
            yield Log(id="update_log", auto_scroll=True)

        yield Static("", id="toolbar")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield DataTable(id="game_table", cursor_type="row", zebra_stripes=True)
            yield ResultPanel(id="result", markup=True)
        yield Footer()

    # ── mount ─────────────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        self.query_one("#game_table", DataTable).add_columns(
            "Date", "Time (ET)", "Matchup", "Away SP", "Home SP", "Status"
        )
        self.query_one("#result", ResultPanel).show_placeholder()
        if self._no_update:
            self._load_games()
        else:
            self._start_update()

    # ── update pipeline ───────────────────────────────────────────────────────
    def _start_update(self) -> None:
        """Kick off the data-refresh worker and stream its log lines into the Log widget."""
        log_q: queue.Queue[str | None] = queue.Queue()
        self._log_q = log_q

        # Start a plain thread that pumps messages from the queue into the Log
        # widget via call_from_thread (safe cross-thread UI update).
        def _drain():
            while True:
                msg = log_q.get()
                if msg is None:
                    break
                self.call_from_thread(self._append_update_log, msg)

        threading.Thread(target=_drain, daemon=True).start()
        self._update_data()

    def _append_update_log(self, msg: str) -> None:
        self.query_one("#update_log", Log).write_line(msg)

    @work(thread=True, name="update_data")
    def _update_data(self) -> int:
        return _run_update(self._log_q)

    # ── schedule loader ───────────────────────────────────────────────────────
    @work(thread=True, name="load_games")
    def _load_games(self) -> list[UpcomingGame]:
        return fetch_upcoming_games(days=2)

    # ── worker state handler ──────────────────────────────────────────────────
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        state = event.state

        # ── update finished → hide panel, start schedule load ────────────────
        if event.worker.name == "update_data" and state == WorkerState.SUCCESS:
            added = event.worker.result or 0
            suffix = f"{added} new game(s) added." if added else "Already up to date."
            self._append_update_log(f"✓ {suffix}")
            # Give the user a moment to read, then collapse the panel
            self.set_timer(1.2, self._hide_update_panel)
            self._load_games()

        elif event.worker.name == "update_data" and state == WorkerState.ERROR:
            self._append_update_log(f"ERROR: {event.worker.error}")
            self.set_timer(2.0, self._hide_update_panel)
            self._load_games()   # still try to load the schedule

        # ── schedule loaded → populate table ─────────────────────────────────
        elif event.worker.name == "load_games" and state == WorkerState.SUCCESS:
            self._games = event.worker.result or []
            self._apply_filter()
            self._refresh_toolbar()

        # ── inference finished → show result ──────────────────────────────────
        elif event.worker.name == "inference" and state == WorkerState.SUCCESS:
            result = event.worker.result
            if result:
                self.query_one("#result", ResultPanel).show_result(result)

        elif event.worker.name == "inference" and state == WorkerState.ERROR:
            self.query_one("#result", ResultPanel).show_error(
                str(event.worker.error)
            )

    def _hide_update_panel(self) -> None:
        self.query_one("#update_panel").add_class("hidden")

    # ── filter helpers ────────────────────────────────────────────────────────
    def action_toggle_filter(self) -> None:
        self._filter = {"all": "today", "today": "tomorrow", "tomorrow": "all"}[self._filter]
        self._apply_filter()
        self._refresh_toolbar()

    def _apply_filter(self) -> None:
        f = self._filter
        if f == "today":
            self._visible = [g for g in self._games if g.date == TODAY]
        elif f == "tomorrow":
            self._visible = [g for g in self._games if g.date == TOMORROW]
        else:
            self._visible = list(self._games)
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        table = self.query_one("#game_table", DataTable)
        table.clear()
        for g in self._visible:
            style = STATUS_STYLE.get(g.status, "")
            score = f"  {g.away_score}–{g.home_score}" if g.away_score is not None else ""
            table.add_row(
                g.date,
                g.game_time_et,
                f"{g.away_team} @ {g.home_team}{score}",
                g.away_sp,
                g.home_sp,
                f"[{style}]{g.status}[/]",
            )

    def _refresh_toolbar(self) -> None:
        labels = {"all": "All Games", "today": "Today Only", "tomorrow": "Tomorrow Only"}
        n, total = len(self._visible), len(self._games)
        self.query_one("#toolbar", Static).update(
            f"[bold]Filter:[/bold] {labels[self._filter]}   "
            f"[bold]Showing:[/bold] {n}/{total} games   "
            f"[dim]T[/dim] filter  ·  [dim]↑↓[/dim] navigate  ·  [dim]Enter/R[/dim] run model"
        )

    # ── inference ─────────────────────────────────────────────────────────────
    def action_run_inference(self) -> None:
        table = self.query_one("#game_table", DataTable)
        if table.cursor_row is None or not self._visible:
            return
        idx = table.cursor_row
        if idx >= len(self._visible):
            return
        game = self._visible[idx]
        self.query_one("#result", ResultPanel).show_loading(game)
        self._infer_game(game)

    @work(thread=True, name="inference")
    def _infer_game(self, game: UpcomingGame) -> dict:
        return _run_inference(game)


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Moneyball TUI — MLB game predictor")
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip the incremental data refresh and go straight to the schedule.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(
            f"ERROR: Model not found at '{MODEL_PATH}'.\n"
            "Run 'python scripts/train.py' first."
        )
        raise SystemExit(1)
    MoneyballApp(no_update=args.no_update).run()


if __name__ == "__main__":
    main()
