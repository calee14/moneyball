import requests
from datetime import datetime


def test_mlb_api():
    # Step 1: Get today's schedule
    # We use sportId=1 to specify Major League Baseball
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching games for {today}...\n")

    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    response = requests.get(schedule_url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error fetching schedule: {response.status_code}")
        return

    schedule_data = response.json()

    # Step 2: Check if there are actually games scheduled today
    if not schedule_data.get("dates") or not schedule_data["dates"][0].get("games"):
        print("No games found for today. (Might be an off day or off-season!)")
        return

    games = schedule_data["dates"][0]["games"]
    print(f"Found {len(games)} game(s) today. Let's look at the first one.\n")

    # Grab the first game in the list
    first_game = games[0]
    game_pk = first_game[
        "gamePk"
    ]  # This is the crucial ID you need for everything else
    away_team = first_game["teams"]["away"]["team"]["name"]
    home_team = first_game["teams"]["home"]["team"]["name"]
    status = first_game["status"]["detailedState"]

    print(f"Selected Game: {away_team} @ {home_team}")
    print(f"Game ID: {game_pk}")
    print(f"Game Status: {status}")
    print("-" * 40)

    # Step 3: Fetch the live feed for that specific game
    # This endpoint contains literally everything: pitch speeds, play-by-play, box scores
    live_feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    live_response = requests.get(live_feed_url)
    live_data = live_response.json()

    # Step 4: Extract some specific game data
    try:
        # Let's get the linescore to see the runs
        linescore = live_data["liveData"]["linescore"]
        inning = linescore.get("currentInningOrdinal", "Pregame")
        away_runs = linescore["teams"]["away"].get("runs", 0)
        home_runs = linescore["teams"]["home"].get("runs", 0)

        print(f"Inning: {inning}")
        print(f"Score: {away_team} {away_runs} - {home_runs} {home_team}")

        # If the game has started, let's grab the last play
        if "currentPlay" in live_data["liveData"]["plays"]:
            last_play = live_data["liveData"]["plays"]["currentPlay"]
            event = last_play["result"].get("event", "No event yet")
            description = last_play["result"].get("description", "No description yet")

            print(f"\nLast Play Result: {event}")
            print(f"Play Description: {description}")

            # If you wanted pitch data, it lives here:
            # print(last_play['playEvents'][-1]['pitchData'])

    except KeyError:
        print(
            "\nCould not parse deep game details. The game might not have started yet, or the JSON structure is missing expected keys."
        )


if __name__ == "__main__":
    test_mlb_api()

