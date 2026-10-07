"""
常用工具函数：
- 合并赔率与比赛数据
- 标准化球队名称（可扩展）
"""
import os
import pandas as pd


def normalize_team_name(name: str) -> str:
    s = str(name).strip()
    # 统一空白，去首尾空格；可以扩展映射表
    return " ".join(s.split())


def merge_odds(matches_df: pd.DataFrame, odds_csv: str) -> pd.DataFrame:
    odds = pd.read_csv(odds_csv, parse_dates=["date"])
    odds["home_team"] = odds["home_team"].apply(normalize_team_name)
    odds["away_team"] = odds["away_team"].apply(normalize_team_name)
    matches_df = matches_df.copy()
    matches_df["home_team"] = matches_df["home_team"].apply(normalize_team_name)
    matches_df["away_team"] = matches_df["away_team"].apply(normalize_team_name)

    merged = matches_df.merge(odds, on=["date", "home_team", "away_team"], how="left")
    return merged
