#!/usr/bin/env python3
"""
命令行工具：把项目流程串起来，便于在终端中一步运行完整流程或子流程。

用法示例：
  python src/cli.py features --raw data/raw/example_matches.csv --out data/processed/processed_matches.csv
  python src/cli.py train --model lightgbm --processed data/processed/processed_matches.csv
  python src/cli.py predict --model lightgbm --input data/processed/processed_matches.csv
  python src/cli.py backtest --pred data/processed/predictions.csv
  python src/cli.py rho-grid

支持子命令：fetch, features, train, train-poisson, predict, backtest, rho-grid, all
"""
import argparse
import os
import sys
import shutil
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, 'src'))

from fetch_fbref import fetch_matches_fbref
from fetch_footballdata_api import fetch_matches_competition
from features import build_features
from model_train import train_lightgbm
from poisson import fit_poisson_pair, predict_probs_for_fixtures
from predict import predict as predict_lgb
from backtest import simple_backtest
from rho_grid_search import run_rho_grid_search
from utils import merge_odds


def ensure_dirs():
    os.makedirs(os.path.join(ROOT, 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'data', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)


def cmd_fetch(args):
    ensure_dirs()
    if args.source == 'fbref':
        df = fetch_matches_fbref(args.league, args.season)
        print(f'Fetched {len(df)} rows from FBref')
    elif args.source == 'footballdata':
        df = fetch_matches_competition(args.competition, args.season)
        print(f'Fetched {len(df)} rows from football-data.org')
    else:
        print('Unknown source')


def cmd_features(args):
    ensure_dirs()
    build_features(args.raw, args.out)


def cmd_train(args):
    ensure_dirs()
    if args.model == 'lightgbm':
        model_path = train_lightgbm(args.processed, model_out=args.model_out)
        print('Saved model to', model_path)
    else:
        print('Unknown model for train')


def cmd_train_poisson(args):
    ensure_dirs()
    out = fit_poisson_pair(args.matches, model_out=args.model_out)
    print('Saved poisson models to', out)


def cmd_predict(args):
    ensure_dirs()
    if args.model == 'lightgbm':
        # use existing predict.py function
        predict_lgb(args.input, args.model_path)
    elif args.model == 'poisson':
        fixtures = pd.read_csv(args.input, parse_dates=['date'])
        res = predict_probs_for_fixtures(args.model_path, fixtures, rho=args.rho, max_goals=args.max_goals)
        out = os.path.join(ROOT, 'data', 'processed', 'predictions.csv')
        res.to_csv(out, index=False)
        print('Saved poisson predictions to', out)
    else:
        print('Unknown model type')


def cmd_backtest(args):
    ensure_dirs()
    hist = simple_backtest(args.predictions, bankroll=args.bankroll, stake_frac=args.stake_frac, min_edge=args.min_edge)
    print('Backtest finished')


def cmd_rho_grid(args):
    ensure_dirs()
    res_df, best = run_rho_grid_search(matches_csv=args.matches, odds_csv=args.odds, rhos=args.rhos,
                                       bankroll=args.bankroll, stake_frac=args.stake_frac, min_edge=args.min_edge,
                                       train_frac=args.train_frac, max_goals=args.max_goals)
    print('Grid search finished; best:', best)


def cmd_all(args):
    ensure_dirs()
    # full pipeline: features -> train both -> predict both -> backtest both
    print('Building features...')
    build_features(args.raw, args.processed)
    print('Training LightGBM...')
    lgb_model = train_lightgbm(args.processed, model_out=args.lgb_model_out)
    print('Training Poisson...')
    p_model = fit_poisson_pair(args.raw, model_out=args.poisson_model_out)
    print('Predicting LightGBM...')
    predict_lgb(args.processed, lgb_model)
    print('Predicting Poisson...')
    fixtures = pd.read_csv(args.processed, parse_dates=['date'])[['date','home_team','away_team']]
    p_res = predict_probs_for_fixtures(p_model, fixtures, rho=args.rho)
    p_out = os.path.join(ROOT, 'data', 'processed', 'predictions_poisson.csv')
    p_res.to_csv(p_out, index=False)
    print('Merging odds and backtesting both...')
    merged_lgb = merge_odds(pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'predictions.csv'), parse_dates=['date']), args.odds)
    merged_lgb.to_csv(os.path.join(ROOT, 'data', 'processed', 'predictions_lgb_with_odds.csv'), index=False)
    simple_backtest(os.path.join(ROOT, 'data', 'processed', 'predictions_lgb_with_odds.csv'), bankroll=args.bankroll, stake_frac=args.stake_frac, min_edge=args.min_edge)
    merged_p = merge_odds(p_res.merge(pd.read_csv(args.raw)[['date','home_team','away_team','home_goals','away_goals']], on=['date','home_team','away_team'], how='left'), args.odds)
    merged_p.to_csv(os.path.join(ROOT, 'data', 'processed', 'predictions_poisson_with_odds.csv'), index=False)
    simple_backtest(os.path.join(ROOT, 'data', 'processed', 'predictions_poisson_with_odds.csv'), bankroll=args.bankroll, stake_frac=args.stake_frac, min_edge=args.min_edge)


def main():
    parser = argparse.ArgumentParser(prog='football-predictor-cli')
    sub = parser.add_subparsers(dest='cmd')

    p_fetch = sub.add_parser('fetch')
    p_fetch.add_argument('--source', choices=['fbref','footballdata'], default='fbref')
    p_fetch.add_argument('--league', default='en/comps/9')
    p_fetch.add_argument('--season', default='2022-2023')
    p_fetch.add_argument('--competition', default='PL')
    p_fetch.set_defaults(func=cmd_fetch)

    p_feat = sub.add_parser('features')
    p_feat.add_argument('--raw', required=True)
    p_feat.add_argument('--out', required=True)
    p_feat.set_defaults(func=cmd_features)

    p_train = sub.add_parser('train')
    p_train.add_argument('--model', choices=['lightgbm'], default='lightgbm')
    p_train.add_argument('--processed', required=True)
    p_train.add_argument('--model-out', default=os.path.join(ROOT, 'models', 'lgb_multi_model.pkl'))
    p_train.set_defaults(func=cmd_train)

    p_trainp = sub.add_parser('train-poisson')
    p_trainp.add_argument('--matches', required=True)
    p_trainp.add_argument('--model-out', default=os.path.join(ROOT, 'models', 'poisson_pair_models.pkl'))
    p_trainp.set_defaults(func=cmd_train_poisson)

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--model', choices=['lightgbm','poisson'], required=True)
    p_pred.add_argument('--input', required=True)
    p_pred.add_argument('--model-path', default=os.path.join(ROOT, 'models', 'lgb_multi_model.pkl'))
    p_pred.add_argument('--rho', type=float, default=0.08)
    p_pred.add_argument('--max-goals', type=int, default=6)
    p_pred.set_defaults(func=cmd_predict)

    p_bt = sub.add_parser('backtest')
    p_bt.add_argument('--predictions', required=True)
    p_bt.add_argument('--bankroll', type=float, default=1000.0)
    p_bt.add_argument('--stake-frac', type=float, default=0.02)
    p_bt.add_argument('--min-edge', type=float, default=0.01)
    p_bt.set_defaults(func=cmd_backtest)

    p_rho = sub.add_parser('rho-grid')
    p_rho.add_argument('--matches', default=os.path.join(ROOT, 'data', 'raw', 'example_matches.csv'))
    p_rho.add_argument('--odds', default=os.path.join(ROOT, 'data', 'raw', 'example_odds.csv'))
    p_rho.add_argument('--rhos', nargs='*', type=float)
    p_rho.add_argument('--bankroll', type=float, default=1000.0)
    p_rho.add_argument('--stake-frac', type=float, default=0.02)
    p_rho.add_argument('--min-edge', type=float, default=0.01)
    p_rho.add_argument('--train-frac', type=float, default=0.7)
    p_rho.add_argument('--max-goals', type=int, default=6)
    p_rho.set_defaults(func=cmd_rho_grid)

    p_all = sub.add_parser('all')
    p_all.add_argument('--raw', default=os.path.join(ROOT, 'data', 'raw', 'example_matches.csv'))
    p_all.add_argument('--processed', default=os.path.join(ROOT, 'data', 'processed', 'processed_matches.csv'))
    p_all.add_argument('--odds', default=os.path.join(ROOT, 'data', 'raw', 'example_odds.csv'))
    p_all.add_argument('--lgb-model-out', default=os.path.join(ROOT, 'models', 'lgb_multi_model.pkl'))
    p_all.add_argument('--poisson-model-out', default=os.path.join(ROOT, 'models', 'poisson_pair_models.pkl'))
    p_all.add_argument('--rho', type=float, default=0.08)
    p_all.add_argument('--bankroll', type=float, default=1000.0)
    p_all.add_argument('--stake-frac', type=float, default=0.02)
    p_all.add_argument('--min-edge', type=float, default=0.01)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    # if rhos empty list -> default handled in function
    if getattr(args, 'rhos', None) is not None and len(args.rhos) == 0:
        args.rhos = None
    args.func(args)


if __name__ == '__main__':
    main()
