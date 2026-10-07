"""
从 FBref 抓取比赛结果表格（演示）。
注意：FBref 页面结构可能更新，可能需要根据页面元素做调整。
"""
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_matches_fbref(league_slug: str, season: str) -> pd.DataFrame:
    url = f"https://fbref.com/{league_slug}/{season}/schedule/{season}-Scores-and-Fixtures"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    tables = pd.read_html(response.text)

    for df in tables:
        cols = [str(c).lower() for c in df.columns]
        if any("home" in c for c in cols) and any("away" in c for c in cols) and any("score" in c for c in cols):
            result = df.copy()
            break
    else:
        raise ValueError("Could not find table. Please inspect the FBref URL and page structure.")

    result.columns = [str(c).strip() for c in result.columns]
    result = result[[c for c in result.columns if c in ["Date", "Home", "Away", "Score"] or c.lower() in ["date", "home", "away", "score"]]]

    # 标准化字段名
    rename_map = {}
    for c in result.columns:
        cl = str(c).lower()
        if "date" in cl:
            rename_map[c] = "date"
        elif "home" in cl and "score" not in cl:
            rename_map[c] = "home_team"
        elif "away" in cl and "score" not in cl:
            rename_map[c] = "away_team"
        elif "score" in cl:
            rename_map[c] = "score"
    result = result.rename(columns=rename_map)

    if "score" in result.columns:
        score_parts = result["score"].astype(str).str.extract(r"(\d+)\s*[-:–]\s*(\d+)")
        result["home_goals"] = pd.to_numeric(score_parts[0], errors="coerce")
        result["away_goals"] = pd.to_numeric(score_parts[1], errors="coerce")

    out = result[["date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["league"] = league_slug
    out["season"] = season

    path = os.path.join(RAW_DIR, f"fbref_{league_slug.replace('/', '_')}_{season}.csv")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows to {path}")
    return out


if __name__ == "__main__":
    # 可根据需要修改联赛/赛季
    df = fetch_matches_fbref("en/comps/9", "2022-2023")
    print(df.head())
