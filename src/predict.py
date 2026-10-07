"""
模型预测脚本。
读取模型 + 特征，并输出 p_H / p_D / p_A
"""
import os
import pandas as pd
import joblib
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")


def predict(input_csv: str, model_path: str):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    df = pd.read_csv(input_csv, parse_dates=["date"]).sort_values("date")
    X = df[feature_cols].fillna(0)
    probs = model.predict(X)
    df[["p_H", "p_D", "p_A"]] = probs

    out_path = os.path.join(PROCESSED_DIR, "predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=os.path.join(MODEL_DIR, "lgb_multi_model.pkl"))
    args = parser.parse_args()
    predict(args.input, args.model)
