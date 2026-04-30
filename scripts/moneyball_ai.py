"""
moneyball_ai.py
---------------
Predicts MLB game winners using a Moneyball-style framework via the Claude API.
Loads CLAUDE_API_KEY from the project .env file automatically.

Example:
    from moneyball_ai import predict_game

    result = predict_game("Athletics", "Royals")
    print(result.pick)
    print(result.full_text)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from scripts/)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096
MAX_WEB_SEARCHES = 8

PROMPT_TEMPLATE = """Predict the winner of {team_a} vs {team_b} on {game_date}. Ignore Vegas odds and betting markets entirely. Use the web_search tool to pull live data on the matchup before analyzing — probable starters, recent bullpen usage, lineup news, weather, and injuries.

Work through these factors in order of impact for this specific game:

1. **Starting pitcher matchups** — both starters' recent form, velocity trends, pitch mix, and command. How does each starter's profile (FB-heavy, GB%, K rate, platoon splits) match up against the *specific* opposing lineup? Flag any hitter who historically crushes this pitcher's primary pitch.

2. **Lineup construction & roster fit** — handedness splits, contact vs power profiles, who's hot in the order. Does either lineup have a structural edge against this starter (e.g., lefty-heavy vs a struggling LHP)?

3. **Bullpen availability & fatigue** — who threw in the last 2-3 games, who's unavailable, leverage arms rested vs burned. If the starter exits in the 5th, who's bridging to the closer for each side?

4. **Travel, rest, schedule context** — getaway games, day-after-night, back end of road trips, time zone changes, day-game-after-night-game lineups (rest days for catchers/regulars).

5. **Injuries & lineup availability** — any regulars out, IL moves, recent call-ups affecting the lineup card.

6. **Park & weather** — park factors (run environment, HR factor, foul territory), wind direction and speed, temperature, humidity. Specifically: does the weather amplify or mute either starter's profile (wind out helps fly-ball pitchers' opponents, etc.)?

7. **Recent form & momentum** — last 10 games run differential, not just W-L. Underlying xwOBA/FIP signals vs surface results to flag teams over- or under-performing.

**Output format:**
- **Pick:** [Team] wins, expected score ~X-Y
- **Confidence:** XX% (with brief justification of why that number, not higher or lower)
- **Key drivers (2-3):** the mechanisms creating the edge — not narrative, actual cause-and-effect
- **Counter-case:** the strongest path to the other team winning, and what you'd need to see in-game to know the pick is wrong

Skip filler. If a factor doesn't meaningfully move the needle for this matchup, say so in one line and move on."""


@dataclass
class PredictionResult:
    """Structured result from a prediction call."""

    team_a: str
    team_b: str
    game_date: str
    full_text: str
    pick: Optional[str]
    confidence: Optional[str]
    raw_response: object  # The full Anthropic Message object, for debugging

    def __str__(self) -> str:
        return self.full_text


def _extract_field(text: str, label: str) -> Optional[str]:
    """Pull a single line value following a bolded label like '**Pick:**'."""
    needle = f"**{label}:**"
    idx = text.find(needle)
    if idx == -1:
        return None
    start = idx + len(needle)
    end = text.find("\n", start)
    snippet = text[start:end] if end != -1 else text[start:]
    return snippet.strip()


def _build_client() -> Anthropic:
    """Build an Anthropic client using CLAUDE_API_KEY from the environment."""
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CLAUDE_API_KEY not found. "
            "Make sure it is set in the .env file at the repo root."
        )
    return Anthropic(api_key=api_key)


def predict_game(
    team_a: str,
    team_b: str,
    game_date: Optional[str] = None,
    *,
    client: Optional[Anthropic] = None,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    max_web_searches: int = MAX_WEB_SEARCHES,
) -> PredictionResult:
    """
    Predict the winner of an MLB game using a Moneyball framework.

    Args:
        team_a: Away team name (e.g. "Athletics").
        team_b: Home team name (e.g. "Royals").
        game_date: Game date as YYYY-MM-DD. Defaults to today.
        client: Optional pre-built Anthropic client.
        model: Override the default model.
        max_tokens: Override the response token limit.
        max_web_searches: Cap on web searches Claude can run per prediction.

    Returns:
        PredictionResult with the full text, parsed pick, parsed confidence,
        and the raw API response for further inspection.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    if client is None:
        client = _build_client()

    prompt = PROMPT_TEMPLATE.format(team_a=team_a, team_b=team_b, game_date=game_date)

    messages = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max_web_searches,
        }
    ]

    # Agentic loop: web_search runs server-side; guard against pause_turn and
    # other multi-step stop reasons.
    response = None
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason in ("end_turn", "stop_sequence", "max_tokens"):
            break

        # For non-terminal stop reasons, append assistant turn and continue.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason not in ("tool_use", "pause_turn"):
            break  # Unknown stop reason — exit defensively.

    full_text = "\n".join(
        block.text for block in response.content if block.type == "text"
    )

    return PredictionResult(
        team_a=team_a,
        team_b=team_b,
        game_date=game_date,
        full_text=full_text,
        pick=_extract_field(full_text, "Pick"),
        confidence=_extract_field(full_text, "Confidence"),
        raw_response=response,
    )
