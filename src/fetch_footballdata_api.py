"""
football-data.org API 示例.
需要在项目根目录创建 .env 文件：
FOOTBALL_DATA_API_TOKEN=your_token_here
"""
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://api.football-data.org/v4/"
TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN")


def fetch_matches_competition(competition_code: str, season: int) -> pd.DataFrame:
    if not TOKEN:
        raise ValueError("Missing FOOTBALL_DATA_API_TOKEN. Create .env with FOOTBALL_DATA_API_TOKEN=...")

    headers = {"X-Auth-Token": TOKEN}
    url = f"{BASE_URL}competitions/{competition_code}/matches?season={season}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    j = resp.json()

    rows = []
    for match in j.get("matches", []):
        rows.append({
            "date": match.get("utcDate"),
            "home_team": match.get("homeTeam", {}).get("name"),
            "away_team": match.get("awayTeam", {}).get("name"),
            "home_goals": match.get("score", {}).get("fullTime", {}).get("home"),
            "away_goals": match.get("score", {}).get("fullTime", {}).get("away"),
            "competition": competition_code,
            "season": season,
            "status": match.get("status"),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    path = os.path.join(RAW_DIR, f"football_data_{competition_code}_{season}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return df


if __name__ == "__main__":
    # 例如：英超（PL）2022赛季
    fetch_matches_competition("PL", 2022)
