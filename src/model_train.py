"""
LightGBM 多分类建模脚本。
输出模型：models/lgb_multi_model.pkl
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
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


def train_lightgbm(processed_csv: str, model_out: str = None):
    df = pd.read_csv(processed_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X = df[FEATURES].fillna(0)
    y = df["y"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbosity": -1,
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            early_stopping_rounds=50,
        )

        val_prob = model.predict(X_val)
        val_loss = log_loss(y_val, val_prob)
        val_pred = val_prob.argmax(axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Fold {fold}: logloss={val_loss:.4f}, acc={val_acc:.4f}")
        fold_scores.append(val_loss)
        best_model = model

    if model_out is None:
        model_out = os.path.join(MODEL_DIR, "lgb_multi_model.pkl")

    joblib.dump({"model": best_model, "features": FEATURES}, model_out)
    print(f"Saved model to {model_out}; mean_cv_logloss={np.mean(fold_scores):.4f}")
    return model_out


if __name__ == "__main__":
    processed_path = os.path.join(PROCESSED_DIR, "processed_matches.csv")
    train_lightgbm(processed_path)
