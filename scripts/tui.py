"""
tui.py — Moneyball TUI (Elo edition)

On launch:
  1. Fetches today's and tomorrow's MLB schedule with probable pitchers.
  2. Browse games, press Enter/A to view the Elo prediction + team comparison.

Controls:
  ↑ / ↓       — navigate games
  Enter / A   — show Elo analysis on selected game
  T           — cycle filter (All / Today / Tomorrow)
  Q / Ctrl+C  — quit
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure the scripts directory is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import Worker, WorkerState

from rich.table import Table
from rich.text import Text

from fetch_upcoming import UpcomingGame, fetch_upcoming_games
from elo_predictor import GamePrediction, TeamContext, get_predictor
import elo_predictor
from refresh import refresh as run_refresh

# ── misc constants ────────────────────────────────────────────────────────────
TODAY = date.today().isoformat()
TOMORROW = (date.today() + timedelta(days=1)).isoformat()

STATUS_STYLE = {
    "Upcoming": "bold cyan",
    "Live": "bold yellow",
    "Final": "dim",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _fmt_diff(a: float, b: float, suffix: str = "", invert: bool = False) -> str:
    """
    Format a comparison delta (b - a). If invert=True, lower is better
    (e.g. fewer bullpen IP is better) so we flip the sign for color logic.
    """
    diff = b - a
    if abs(diff) < 0.05:
        color = "dim"
        arrow = "≈"
    else:
        better_for_b = (diff > 0) ^ invert
        color = "green" if better_for_b else "red"
        arrow = "▲" if diff > 0 else "▼"
    return f"[{color}]{arrow} {abs(diff):.1f}{suffix}[/{color}]"


# ── comparison panel ──────────────────────────────────────────────────────────
class ComparisonPanel(Vertical):
    """
    Right-hand panel: Elo header + win prob bar + side-by-side team stats.
    """

    DEFAULT_CSS = """
    ComparisonPanel {
        border: solid $accent;
        height: 100%;
        width: 1fr;
        padding: 1;
    }
    ComparisonPanel > #cp_header {
        padding: 0 1 1 1;
        height: auto;
        background: $surface;
        border-bottom: solid $panel;
    }
    ComparisonPanel > #cp_winprob {
        padding: 1;
        height: auto;
    }
    ComparisonPanel > #cp_compare {
        padding: 0 1;
        height: 1fr;
        overflow-y: auto;
    }
    ComparisonPanel > #cp_notes {
        padding: 1 1 0 1;
        height: auto;
        background: $surface;
        border-top: solid $accent;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="cp_header", markup=True)
        yield Static("", id="cp_winprob", markup=True)
        yield Static("", id="cp_compare", markup=True)
        yield Static("", id="cp_notes", markup=True)

    def _set(self, sel: str, content: object) -> None:
        self.query_one(sel, Static).update(content)

    # ── public API ────────────────────────────────────────────────────────────
    def show_placeholder(self) -> None:
        self._set(
            "#cp_header",
            "[dim]Select a game and press Enter / A to view Elo analysis.[/dim]",
        )
        self._set("#cp_winprob", "")
        self._set("#cp_compare", "")
        self._set("#cp_notes", "")

    def show_loading(self, game: UpcomingGame) -> None:
        self._set(
            "#cp_header",
            f"[bold yellow]Computing…[/bold yellow]  "
            f"[cyan]{game.away_team}[/cyan]  @  [cyan]{game.home_team}[/cyan]",
        )
        self._set("#cp_winprob", "")
        self._set("#cp_compare", "")
        self._set("#cp_notes", "")

    def show_error(self, msg: str) -> None:
        self._set("#cp_header", "[bold red]Error[/bold red]")
        self._set("#cp_winprob", "")
        self._set("#cp_compare", f"[red]{msg}[/red]")
        self._set("#cp_notes", "")

    def show_refresh_progress(self, log_text: str) -> None:
        self._set("#cp_header", "[bold cyan]Refreshing data + ratings…[/bold cyan]")
        self._set("#cp_winprob", "")
        self._set("#cp_compare", f"[dim]{log_text}[/dim]")
        self._set(
            "#cp_notes",
            "[dim]This usually takes 30–60 seconds. "
            "App will reload schedule when complete.[/dim]",
        )

    def show_prediction(self, pred: GamePrediction, game: UpcomingGame) -> None:
        self._set("#cp_header", self._build_header(pred, game))
        self._set("#cp_winprob", self._build_winprob(pred))
        self._set("#cp_compare", self._build_comparison(pred))
        self._set("#cp_notes", self._build_notes(pred))

    # ── builders ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build_header(pred: GamePrediction, game: UpcomingGame) -> str:
        a, h = pred.away, pred.home
        return (
            f"[bold]{a.team_name}[/bold] (#{a.rank}, Elo {a.elo:.0f})  "
            f"[dim]@[/dim]  "
            f"[bold]{h.team_name}[/bold] (#{h.rank}, Elo {h.elo:.0f})\n"
            f"[dim]{game.away_sp} vs {game.home_sp}  ·  "
            f"{game.game_time_et}  ·  Elo gap: "
            f"{'+' if pred.elo_diff > 0 else ''}{pred.elo_diff:.0f} (home)[/dim]"
        )

    @staticmethod
    def _build_winprob(pred: GamePrediction) -> Text:
        a_pct = pred.p_away_win
        h_pct = pred.p_home_win
        bar_w = 40
        a_blocks = round(a_pct * bar_w)
        bar = Text("█" * bar_w)
        bar.stylize("cyan", 0, a_blocks)
        bar.stylize("magenta", a_blocks, bar_w)

        team_col = 22
        text = Text()
        text.append("Win Probability (pure team Elo)\n\n", style="bold")
        text.append(Text(pred.away.team_name.ljust(team_col), style="cyan"))
        text.append(" ")
        text.append(bar)
        text.append(" ")
        text.append(Text(pred.home.team_name + "\n", style="magenta"))

        text.append(Text(_fmt_pct(a_pct).rjust(team_col), style="cyan"))
        text.append(" ")
        text.append(" " * bar_w)
        text.append(" ")
        text.append(Text(_fmt_pct(h_pct) + "\n", style="magenta"))

        favored = pred.home if pred.pick_team == "home" else pred.away
        favored_pct = max(a_pct, h_pct)

        text.append("\n")
        text.append("Pick: ")
        text.append(favored.team_name, style="bold green")
        text.append(
            f" ({pred.pick_confidence}, {_fmt_pct(favored_pct)} vs {_fmt_pct(1 - favored_pct)})"
        )
        return text

    @staticmethod
    def _build_comparison(pred: GamePrediction) -> Table:
        a, h = pred.away, pred.home

        def _safe(val: float | None, fmt: str = "{:.1f}", suffix: str = "") -> str:
            if val is None or _isnan(val):
                return "n/a"
            return (fmt + suffix).format(val)

        table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
        table.add_column("", style="dim", no_wrap=True)
        table.add_column(a.team_name, style="cyan", no_wrap=True, justify="left")
        table.add_column(h.team_name, style="magenta", no_wrap=True, justify="left")
        table.add_column("Edge (home)", style="bold", no_wrap=True, justify="left")

        elo_edge = _fmt_diff(a.elo, h.elo)
        l10_w_edge = _fmt_diff(a.last_10_wins, h.last_10_wins)
        rd_edge = _fmt_diff(a.last_10_run_diff, h.last_10_run_diff)
        rest_edge = _fmt_diff(a.days_rest, h.days_rest, "d")
        pen_edge = _fmt_diff(a.bullpen_ip_last3, h.bullpen_ip_last3, " IP", invert=True)

        l10_a = f"{a.last_10_wins:.0f}-{a.last_10_losses:.0f}"
        l10_h = f"{h.last_10_wins:.0f}-{h.last_10_losses:.0f}"
        rd_a = f"{'+' if a.last_10_run_diff > 0 else ''}{a.last_10_run_diff}"
        rd_h = f"{'+' if h.last_10_run_diff > 0 else ''}{h.last_10_run_diff}"
        ar_a = f"{a.away_record[0]}-{a.away_record[1]} (away)"
        hr_h = f"{h.home_record[0]}-{h.home_record[1]} (home)"
        rest_a = _safe(a.days_rest, "{:.1f}", "d")
        rest_h = _safe(h.days_rest, "{:.1f}", "d")
        pen_a = _safe(a.bullpen_ip_last3, "{:.1f}", " IP")
        pen_h = _safe(h.bullpen_ip_last3, "{:.1f}", " IP")

        def _sp_era(ctx: TeamContext) -> str:
            if ctx.starter_era_recent is not None:
                return f"{ctx.starter_era_recent:.2f} ERA (L{ctx.starter_era_n})"
            return "n/a"

        sp_a = _sp_era(a)
        sp_h = _sp_era(h)
        sp_edge = (
            _fmt_diff(a.starter_era_recent or 0.0, h.starter_era_recent or 0.0, invert=True)
            if (a.starter_era_recent is not None and h.starter_era_recent is not None)
            else ""
        )

        table.add_row("Elo rating", f"{a.elo:.0f} (#{a.rank})", f"{h.elo:.0f} (#{h.rank})", Text.from_markup(elo_edge))
        table.add_row("Last 10 record", l10_a, l10_h, Text.from_markup(l10_w_edge))
        table.add_row("Last 10 run diff", rd_a, rd_h, Text.from_markup(rd_edge))
        table.add_row("Season record", ar_a, hr_h, "")
        table.add_row("Days rest", rest_a, rest_h, Text.from_markup(rest_edge))
        table.add_row("Bullpen IP last 3", pen_a, pen_h, Text.from_markup(pen_edge))
        table.add_row("SP recent ERA", sp_a, sp_h, Text.from_markup(sp_edge))

        return table

    @staticmethod
    def _build_notes(pred: GamePrediction) -> str:
        if not pred.notes:
            return "[dim]  No flags — model & context are aligned.[/dim]"
        body = "\n".join(f"  {n}" for n in pred.notes)
        return f"[bold]Flags worth noting[/bold]\n{body}"


# ── helpers ───────────────────────────────────────────────────────────────────
def _isnan(x) -> bool:
    try:
        return x != x  # NaN != NaN
    except Exception:
        return False


# ── main app ──────────────────────────────────────────────────────────────────
class MoneyballApp(App):
    TITLE = "Moneyball — MLB Elo Predictor"
    SUB_TITLE = "Today & Tomorrow"

    CSS = """
    Screen { layout: vertical; }

    #toolbar {
        height: 1;
        background: $boost;
        padding: 0 1;
        color: $text-muted;
    }

    #main { layout: horizontal; height: 1fr; }
    #left  { width: 2fr; height: 100%; border: solid $panel; padding: 0 1; }
    #right { width: 3fr; height: 100%; }

    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "toggle_filter", "Filter"),
        Binding("a", "run_analysis", "Elo Analysis"),
        Binding("enter", "run_analysis", "Elo Analysis", show=False),
        Binding("r", "refresh", "Refresh data"),
    ]

    _filter: reactive[str] = reactive("all")
    _games: list[UpcomingGame] = []
    _visible: list[UpcomingGame] = []
    _refreshing: bool = False

    # ── compose ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="toolbar")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield DataTable(id="game_table", cursor_type="row", zebra_stripes=True)
            yield ComparisonPanel(id="result")
        yield Footer()

    # ── mount ─────────────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        table = self.query_one("#game_table", DataTable)
        table.add_column("Date", width=12)
        table.add_column("Time (ET)", width=10)
        table.add_column("Matchup", width=30)
        table.add_column("Away SP", width=20)
        table.add_column("Home SP", width=20)
        table.add_column("Status", width=10)
        self.query_one("#result", ComparisonPanel).show_placeholder()
        # Warm up the predictor in a background thread so first analysis is instant
        self._warmup_predictor()
        self._load_games()

    @work(thread=True, name="warmup")
    def _warmup_predictor(self) -> None:
        try:
            get_predictor()
        except Exception:
            pass  # Surfaced when user actually clicks a game

    # ── schedule loader ───────────────────────────────────────────────────────
    @work(thread=True, name="load_games")
    def _load_games(self) -> list[UpcomingGame]:
        return fetch_upcoming_games(days=2)

    # ── worker state handler ──────────────────────────────────────────────────
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        state = event.state
        name = event.worker.name

        if name == "load_games" and state == WorkerState.SUCCESS:
            self._games = event.worker.result or []
            self._apply_filter()
            self._refresh_toolbar()
        elif name == "load_games" and state == WorkerState.ERROR:
            self.query_one("#result", ComparisonPanel).show_error(
                f"Failed to load schedule: {event.worker.error}"
            )

        elif name == "refresh" and state == WorkerState.SUCCESS:
            r = event.worker.result or {}
            panel = self.query_one("#result", ComparisonPanel)
            self._on_refresh_progress(
                f"\n✓ Refresh complete: {r.get('new_games', 0)} new games, "
                f"{r.get('total_games', 0)} total. Reloading schedule..."
            )
            self._refreshing = False
            # Re-fetch upcoming games (today/tomorrow may have changed)
            self._load_games()
            # Show placeholder once we're done
            self.set_timer(
                2.0, lambda: panel.show_placeholder() if not self._refreshing else None
            )
        elif name == "refresh" and state == WorkerState.ERROR:
            self._refreshing = False
            self.query_one("#result", ComparisonPanel).show_error(
                f"Refresh failed: {event.worker.error}"
            )

    # ── filter helpers ────────────────────────────────────────────────────────
    def action_toggle_filter(self) -> None:
        self._filter = {"all": "today", "today": "tomorrow", "tomorrow": "all"}[
            self._filter
        ]
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
            score = (
                f"  {g.away_score}–{g.home_score}" if g.away_score is not None else ""
            )
            status_cell = Text(g.status, style=style) if style else g.status
            table.add_row(
                g.date,
                g.game_time_et,
                f"{g.away_team} @ {g.home_team}{score}",
                g.away_sp,
                g.home_sp,
                status_cell,
            )

    def _refresh_toolbar(self) -> None:
        labels = {
            "all": "All Games",
            "today": "Today Only",
            "tomorrow": "Tomorrow Only",
        }
        n, total = len(self._visible), len(self._games)
        self.query_one("#toolbar", Static).update(
            f"[bold]Filter:[/bold] {labels[self._filter]}   "
            f"[bold]Showing:[/bold] {n}/{total} games   "
            f"[dim]T[/dim] filter  ·  [dim]↑↓[/dim] navigate  ·  "
            f"[dim]Enter/A[/dim] Elo analysis  ·  [dim]R[/dim] refresh"
        )

    # ── Elo analysis (synchronous — instant) ──────────────────────────────────
    def action_run_analysis(self) -> None:
        if self._refreshing:
            return  # Don't analyze while refresh is mid-flight
        table = self.query_one("#game_table", DataTable)
        if table.cursor_row is None or not self._visible:
            return
        idx = table.cursor_row
        if idx >= len(self._visible):
            return
        game = self._visible[idx]
        panel = self.query_one("#result", ComparisonPanel)

        try:
            predictor = get_predictor()
            pred = predictor.predict(
                away_team=game.away_team,
                home_team=game.home_team,
                away_sp_id=game.away_sp_id,
                home_sp_id=game.home_sp_id,
                away_sp=game.away_sp,
                home_sp=game.home_sp,
            )
            panel.show_prediction(pred, game)
        except Exception as e:
            panel.show_error(f"{type(e).__name__}: {e}")

    # ── Refresh ───────────────────────────────────────────────────────────────
    def action_refresh(self) -> None:
        """Refresh data + ratings, then reload predictor and upcoming games."""
        if self._refreshing:
            return
        self._refreshing = True
        panel = self.query_one("#result", ComparisonPanel)
        panel.show_refresh_progress("Starting refresh...")
        self._refresh_progress_lines: list[str] = []
        self._do_refresh()

    def _on_refresh_progress(self, line: str) -> None:
        """Called from the refresh worker via call_from_thread."""
        self._refresh_progress_lines.append(line)
        # Keep only the last 12 lines so the panel doesn't run off-screen
        tail = self._refresh_progress_lines[-12:]
        self.query_one("#result", ComparisonPanel).show_refresh_progress(
            "\n".join(tail)
        )

    @work(thread=True, name="refresh")
    def _do_refresh(self) -> dict:
        # Bridge thread-side progress back to the main loop
        def progress(line: str) -> None:
            self.call_from_thread(self._on_refresh_progress, str(line))

        result = run_refresh(progress=progress)
        # Force the predictor singleton to rebuild on next access
        elo_predictor._predictor = None
        return result


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    MoneyballApp().run()


if __name__ == "__main__":
    main()
