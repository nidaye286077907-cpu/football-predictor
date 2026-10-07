"""
rho_grid_search.py

在训练好的 Poisson 模型上对 rho 做网格搜索，并对每个 rho 运行回测，输出回测统计（ROI、最终余额、下注次数），并保存结果与图表。

使用：
python src/rho_grid_search.py

或者在 Notebook 中导入 run_rho_grid_search 并传参调用。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.poisson import fit_poisson_pair, predict_probs_for_fixtures
from src.utils import merge_odds
from src.backtest import simple_backtest

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, 'data', 'raw')
DATA_PROC = os.path.join(ROOT, 'data', 'processed')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(DATA_PROC, exist_ok=True)


def run_rho_grid_search(matches_csv: str = os.path.join(DATA_RAW, 'example_matches.csv'),
                        odds_csv: str = os.path.join(DATA_RAW, 'example_odds.csv'),
                        rhos: list = None,
                        bankroll: float = 1000.0,
                        stake_frac: float = 0.02,
                        min_edge: float = 0.01,
                        train_frac: float = 0.7,
                        max_goals: int = 6):
    if rhos is None:
        rhos = list(np.linspace(0.0, 0.2, 21))
    # load matches
    df = pd.read_csv(matches_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    # save train/test to temporary csv
    train_csv = os.path.join(DATA_PROC, 'rho_train.csv')
    test_csv = os.path.join(DATA_PROC, 'rho_test.csv')
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # fit poisson on train
    model_path = fit_poisson_pair(train_csv)

    results = []
    for rho in rhos:
        print(f"Testing rho={rho:.4f} ...")
        # predict lambdas and outcome probs on test set
        fixtures = test_df[['date','home_team','away_team']].copy()
        probs = predict_probs_for_fixtures(model_path, fixtures, rho=rho, max_goals=max_goals)
        # merge back actual outcomes and odds
        merged = probs.merge(test_df[['date','home_team','away_team','home_goals','away_goals']], on=['date','home_team','away_team'], how='left')
        # compute outcome label
        merged['home_goal_diff'] = merged['home_goals'] - merged['away_goals']
        merged['outcome'] = merged['home_goal_diff'].apply(lambda x: 'H' if x>0 else ('A' if x<0 else 'D'))

        merged = merge_odds(merged, odds_csv)
        # ensure columns odd_h etc exist
        # save predictions file
        pred_path = os.path.join(DATA_PROC, f'predictions_rho_{rho:.3f}.csv')
        merged.to_csv(pred_path, index=False)

        # run backtest but capture history
        hist = simple_backtest(pred_path, bankroll=bankroll, stake_frac=stake_frac, min_edge=min_edge)
        # simple_backtest prints metrics; hist contains balance over time
        final_balance = hist['balance'].dropna().iloc[-1] if not hist.empty else bankroll
        total_return = final_balance - bankroll
        roi = total_return / bankroll
        num_bets = hist[hist['stake']>0].shape[0]

        # save individual balance plot
        plt.figure(figsize=(8,4))
        if 'date' in hist.columns:
            plt.plot(pd.to_datetime(hist['date']), hist['balance'].ffill())
            plt.title(f'Balance over time (rho={rho:.3f})')
            plt.xlabel('Date')
            plt.ylabel('Balance')
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(DATA_PROC, f'backtest_balance_rho_{rho:.3f}.png')
            plt.savefig(plot_path)
            plt.close()
        else:
            plot_path = None

        results.append({'rho': rho, 'final_balance': final_balance, 'total_return': total_return, 'roi': roi, 'num_bets': num_bets, 'pred_csv': pred_path, 'plot': plot_path})

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('roi', ascending=False).reset_index(drop=True)
    out_csv = os.path.join(DATA_PROC, 'rho_grid_results.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"Saved grid results to {out_csv}")

    best = res_df.iloc[0].to_dict()
    print("Best rho:", best)
    # copy best predictions to predictions.csv for convenience
    best_pred = best.get('pred_csv')
    if best_pred:
        best_out = os.path.join(DATA_PROC, 'predictions_best_rho.csv')
        pd.read_csv(best_pred).to_csv(best_out, index=False)
        print(f"Saved best predictions to {best_out}")
    return res_df, best


if __name__ == '__main__':
    run_rho_grid_search()
