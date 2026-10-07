"""
数据特征工程脚本：
- 近 5 场球队滚动表现
- 近 5 场主客场比较
- 简单 Elo 评分
- 构建模型输入特征
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_matches(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["date"], infer_datetime_format=True)
    if "utcDate" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["utcDate"])
    return df


def add_result_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["home_goals", "away_goals"]:
        if col not in df.columns:
            df[col] = np.nan
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["home_goal_diff"] = df["home_goals"] - df["away_goals"]
    df["outcome"] = df["home_goal_diff"].apply(lambda x: "H" if x > 0 else ("A" if x < 0 else "D"))
    return df


def rolling_team_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    home = df[["date", "home_team", "home_goals", "away_goals"]].rename(columns={
        "home_team": "team",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
    })
    away = df[["date", "away_team", "away_goals", "home_goals"]].rename(columns={
        "away_team": "team",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
    })
    team_stats = pd.concat([home, away], ignore_index=True)
    team_stats = team_stats.sort_values(["team", "date"]).reset_index(drop=True)
    team_stats["result"] = np.where(team_stats["goals_for"] > team_stats["goals_against"], 1,
                                   np.where(team_stats["goals_for"] < team_stats["goals_against"], -1, 0))

    team_stats["gf_rolling"] = team_stats.groupby("team")["goals_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    team_stats["ga_rolling"] = team_stats.groupby("team")["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    team_stats["form_rolling"] = team_stats.groupby("team")["result"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum())

    home_features = team_stats[["team", "date", "gf_rolling", "ga_rolling", "form_rolling"]].rename(columns={
        "team": "home_team",
        "gf_rolling": "home_gf_roll",
        "ga_rolling": "home_ga_roll",
        "form_rolling": "home_form",
    })
    away_features = team_stats[["team", "date", "gf_rolling", "ga_rolling", "form_rolling"]].rename(columns={
        "team": "away_team",
        "gf_rolling": "away_gf_roll",
        "ga_rolling": "away_ga_roll",
        "form_rolling": "away_form",
    })

    df = df.merge(home_features, on=["home_team", "date"], how="left")
    df = df.merge(away_features, on=["away_team", "date"], how="left")

    for col in ["home_gf_roll", "home_ga_roll", "home_form", "away_gf_roll", "away_ga_roll", "away_form"]:
        df[col] = df[col].fillna(df[col].mean())

    df["diff_gf_roll"] = df["home_gf_roll"] - df["away_gf_roll"]
    df["diff_ga_roll"] = df["home_ga_roll"] - df["away_ga_roll"]
    df["diff_form"] = df["home_form"] - df["away_form"]
    return df


def add_elo(df: pd.DataFrame, k: float = 20.0, init_score: int = 1500) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    elo_map = {team: init_score for team in set(df["home_team"]).union(set(df["away_team"]))}
    home_elo_pre = []
    away_elo_pre = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        h_elo = elo_map.get(home_team, init_score)
        a_elo = elo_map.get(away_team, init_score)
        home_elo_pre.append(h_elo)
        away_elo_pre.append(a_elo)

        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400.0))
        if row["home_goal_diff"] > 0:
            actual_h = 1.0
        elif row["home_goal_diff"] == 0:
            actual_h = 0.5
        else:
            actual_h = 0.0

        elo_map[home_team] = h_elo + k * (actual_h - exp_h)
        elo_map[away_team] = a_elo + k * ((1.0 - actual_h) - (1.0 - exp_h))

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    return df


def build_features(raw_csv_path: str, out_path: str):
    df = load_raw_matches(raw_csv_path)
    df = add_result_labels(df)
    df = rolling_team_stats(df, window=5)
    df = add_elo(df)

    feature_cols = [
        "home_gf_roll",
        "home_ga_roll",
        "away_gf_roll",
        "away_ga_roll",
        "diff_gf_roll",
        "diff_ga_roll",
        "diff_form",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
    ]

    df["y"] = df["outcome"].map({"H": 0, "D": 1, "A": 2})
    final = df[["date", "home_team", "away_team", "home_goals", "away_goals", "outcome", "y"] + feature_cols].copy()
    final.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return final


if __name__ == "__main__":
    raw = os.path.join(RAW_DIR, "example_matches.csv")
    out = os.path.join(PROCESSED_DIR, "processed_matches.csv")
    build_features(raw, out)
