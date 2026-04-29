import requests
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

session = requests.Session()


def get_season_game_ids(year):
    print(f"Fetching schedule for {year}...")
    start_date = f"{year}-03-20"
    end_date = f"{year}-11-05"
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&gameType=R"

    response = session.get(schedule_url)
    if response.status_code != 200:
        print(f"Error fetching schedule: {response.status_code}")
        return []

    data = response.json()
    game_ids = []
    if "dates" in data:
        for date_obj in data["dates"]:
            for game in date_obj["games"]:
                if game["status"]["statusCode"] in ["F", "O"]:
                    game_ids.append(game["gamePk"])

    print(f"Found {len(game_ids)} completed regular season games for {year}.")
    return game_ids


def fetch_game(game_pk):
    try:
        live_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        response = session.get(live_url, timeout=30)
        game_data = response.json()

        game_date = game_data["gameData"]["datetime"].get("officialDate", "Unknown")
        teams = game_data["gameData"]["teams"]
        away_team = teams["away"]["name"]
        home_team = teams["home"]["name"]

        linescore = game_data["liveData"]["linescore"]
        away_score = linescore["teams"]["away"].get("runs", 0)
        home_score = linescore["teams"]["home"].get("runs", 0)
        home_win = 1 if home_score > away_score else 0

        probable = game_data["gameData"].get("probablePitchers", {})
        away_sp_id = probable.get("away", {}).get("id")
        home_sp_id = probable.get("home", {}).get("id")
        away_sp_name = probable.get("away", {}).get("fullName", "Unknown")
        home_sp_name = probable.get("home", {}).get("fullName", "Unknown")

        boxscore = game_data["liveData"]["boxscore"]["teams"]
        away_bat = boxscore["away"]["teamStats"]["batting"]
        home_bat = boxscore["home"]["teamStats"]["batting"]
        away_pitch = boxscore["away"]["teamStats"]["pitching"]
        home_pitch = boxscore["home"]["teamStats"]["pitching"]

        # UPGRADE: Now pulling Strikeouts (K), Walks (BB), and Home Runs (HR)
        away_sp_er, away_sp_ip, away_sp_k, away_sp_bb, away_sp_hr = 0, "0.0", 0, 0, 0
        if away_sp_id:
            s = (
                boxscore["away"]["players"]
                .get(f"ID{away_sp_id}", {})
                .get("stats", {})
                .get("pitching", {})
            )
            away_sp_er = s.get("earnedRuns", 0)
            away_sp_ip = s.get("inningsPitched", "0.0")
            away_sp_k = s.get("strikeOuts", 0)
            away_sp_bb = s.get("baseOnBalls", 0)
            away_sp_hr = s.get("homeRuns", 0)

        home_sp_er, home_sp_ip, home_sp_k, home_sp_bb, home_sp_hr = 0, "0.0", 0, 0, 0
        if home_sp_id:
            s = (
                boxscore["home"]["players"]
                .get(f"ID{home_sp_id}", {})
                .get("stats", {})
                .get("pitching", {})
            )
            home_sp_er = s.get("earnedRuns", 0)
            home_sp_ip = s.get("inningsPitched", "0.0")
            home_sp_k = s.get("strikeOuts", 0)
            home_sp_bb = s.get("baseOnBalls", 0)
            home_sp_hr = s.get("homeRuns", 0)

        return [
            game_pk,
            game_date,
            away_team,
            home_team,
            away_sp_name,
            home_sp_name,
            away_score,
            home_score,
            away_bat.get("atBats", 0),
            away_bat.get("hits", 0),
            away_bat.get("doubles", 0),
            away_bat.get("triples", 0),
            away_bat.get("homeRuns", 0),
            away_bat.get("baseOnBalls", 0),
            away_bat.get("hitByPitch", 0),
            away_bat.get("sacFlies", 0),
            away_bat.get("strikeOuts", 0),
            home_bat.get("atBats", 0),
            home_bat.get("hits", 0),
            home_bat.get("doubles", 0),
            home_bat.get("triples", 0),
            home_bat.get("homeRuns", 0),
            home_bat.get("baseOnBalls", 0),
            home_bat.get("hitByPitch", 0),
            home_bat.get("sacFlies", 0),
            home_bat.get("strikeOuts", 0),
            away_pitch.get("earnedRuns", 0),
            away_pitch.get("inningsPitched", "0.0"),
            home_pitch.get("earnedRuns", 0),
            home_pitch.get("inningsPitched", "0.0"),
            away_sp_er,
            away_sp_ip,
            away_sp_k,
            away_sp_bb,
            away_sp_hr,  # Added BB and HR
            home_sp_er,
            home_sp_ip,
            home_sp_k,
            home_sp_bb,
            home_sp_hr,  # Added BB and HR
            home_win,
        ]
    except Exception as e:
        return None


def build_dataset(years, output_filename, max_workers=10):
    all_game_ids = []
    for year in years:
        all_game_ids.extend(get_season_game_ids(year))

    print(f"\nTotal games to process: {len(all_game_ids)}")

    # Updated Headers
    header = [
        "Game_ID",
        "Date",
        "Away_Team",
        "Home_Team",
        "Away_SP",
        "Home_SP",
        "Away_Score",
        "Home_Score",
        "Away_AB",
        "Away_Hits",
        "Away_2B",
        "Away_3B",
        "Away_HR",
        "Away_BB",
        "Away_HBP",
        "Away_SF",
        "Away_K",
        "Home_AB",
        "Home_Hits",
        "Home_2B",
        "Home_3B",
        "Home_HR",
        "Home_BB",
        "Home_HBP",
        "Home_SF",
        "Home_K",
        "Away_Team_ER",
        "Away_Team_IP",
        "Home_Team_ER",
        "Home_Team_IP",
        "Away_SP_ER",
        "Away_SP_IP",
        "Away_SP_K",
        "Away_SP_BB",
        "Away_SP_HR",
        "Home_SP_ER",
        "Home_SP_IP",
        "Home_SP_K",
        "Home_SP_BB",
        "Home_SP_HR",
        "Home_Win",
    ]

    write_lock = Lock()
    processed = 0

    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_game, pk): pk for pk in all_game_ids}
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    with write_lock:
                        writer.writerow(row)
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(all_game_ids)} games...")

    print(f"\nFinished! Data saved to {output_filename}")


if __name__ == "__main__":
    seasons_to_pull = [2024, 2025, 2026]
    output_file = "data/mlb_historical_games.csv"
    build_dataset(seasons_to_pull, output_file, max_workers=15)
