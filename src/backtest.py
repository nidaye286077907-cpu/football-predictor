"""
简单回测脚本：
根据模型概率 + 历史赔率，选择 edge 最大的选项。
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
PRED_PATH = os.path.join(ROOT, "data", "processed", "predictions.csv")


def simple_backtest(predictions_csv: str, bankroll: float = 10000.0, stake_frac: float = 0.01, min_edge: float = 0.02):
    df = pd.read_csv(predictions_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df[df[["odd_h", "odd_d", "odd_a"]].notna().all(axis=1)].copy()

    balance = bankroll
    history = []

    for _, row in df.iterrows():
        probs = {"H": row["p_H"], "D": row["p_D"], "A": row["p_A"]}
        odds = {"H": row["odd_h"], "D": row["odd_d"], "A": row["odd_a"]}
        implied = {k: 1.0 / float(v) for k, v in odds.items()}
        edges = {k: probs[k] - implied[k] for k in ["H", "D", "A"]}
        selected = max(edges, key=edges.get)

        if edges[selected] < min_edge:
            history.append({"date": row["date"], "bet": None, "stake": 0, "payout": 0.0, "balance": balance})
            continue

        stake = bankroll * stake_frac
        actual = row["outcome"]
        if selected == actual:
            payout = stake * (odds[selected] - 1.0)
            balance += payout
        else:
            payout = -stake
            balance += payout

        history.append({
            "date": row["date"],
            "bet": selected,
            "stake": stake,
            "payout": payout,
            "balance": balance,
            "edge": edges[selected],
        })

    hist = pd.DataFrame(history)
    total_return = balance - bankroll
    roi = total_return / bankroll
    bets = hist[hist["stake"] > 0].shape[0]

    print(f"Bankroll={bankroll}, Final balance={balance:.2f}, Total return={total_return:.2f}, ROI={roi:.4f}, Bets={bets}")

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(hist["date"]), hist["balance"].values)
    plt.title("Backtest Balance")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "backtest_balance.png"))
    print("Saved plot to backtest_balance.png")

    return hist


if __name__ == "__main__":
    simple_backtest(PRED_PATH)
