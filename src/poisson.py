"""
Poisson / Bivariate Poisson prediction module

功能：
- 基于球队进球数据拟合两个 Poisson 回归模型：主队进球和客队进球
- 使用球队攻击/防守虚拟变量（attack_home, defense_away）和主场优势
- 给定两个边缘 lambda_h, lambda_a 和共享成分比例 rho，计算二元泊松联合分布（通过共享 Poisson 分解）并得到胜平负概率

注意：此实现使用简单的 dummy 编码与 PoissonRegressor（sklearn）。可作为入门与基线模型。
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _prepare_team_dummies(df: pd.DataFrame):
    """构造 attack/defense dummies。返回带有列前缀的 DataFrame 和队名列表"""
    teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    attack_cols = [f"att_{t}" for t in teams]
    defense_cols = [f"def_{t}" for t in teams]

    # 初始化为 0
    attacks = pd.DataFrame(0, index=df.index, columns=attack_cols)
    defs = pd.DataFrame(0, index=df.index, columns=defense_cols)

    for i, r in df.iterrows():
        attacks.at[i, f"att_{r['home_team']}"] = 1
        defs.at[i, f"def_{r['away_team']}"] = 1
    return attacks, defs, teams


def fit_poisson_pair(matches_csv: str, model_out: str = None, alpha: float = 1e-6):
    """
    训练两个 Poisson 模型：主队进球与客队进球。
    输入CSV应包含：date, home_team, away_team, home_goals, away_goals
    返回保存模型路径（包含两个模型与元信息）
    """
    df = pd.read_csv(matches_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 为主队建 attack(home)/defense(away) dummies
    att_home, def_away, teams = _prepare_team_dummies(df)
    X_home = pd.concat([att_home, def_away], axis=1)
    y_home = df['home_goals'].astype(float).fillna(0)

    # 为客队建 attack(away)/defense(home) dummies
    att_away = pd.DataFrame(0, index=df.index, columns=[f"att_{t}" for t in teams])
    def_home = pd.DataFrame(0, index=df.index, columns=[f"def_{t}" for t in teams])
    for i, r in df.iterrows():
        att_away.at[i, f"att_{r['away_team']}"] = 1
        def_home.at[i, f"def_{r['home_team']}"] = 1
    X_away = pd.concat([att_away, def_home], axis=1)
    y_away = df['away_goals'].astype(float).fillna(0)

    # Add intercept is handled by PoissonRegressor via fit_intercept=True
    model_home = PoissonRegressor(alpha=alpha, max_iter=300).fit(X_home, y_home)
    model_away = PoissonRegressor(alpha=alpha, max_iter=300).fit(X_away, y_away)

    bundle = {
        'model_home': model_home,
        'model_away': model_away,
        'teams': teams,
        'feature_home_cols': X_home.columns.tolist(),
        'feature_away_cols': X_away.columns.tolist()
    }
    out = model_out or os.path.join(MODEL_DIR, 'poisson_pair_models.pkl')
    joblib.dump(bundle, out)
    print(f"Saved poisson pair models to {out}")
    return out


def _build_feature_row(home_team: str, away_team: str, teams: list):
    att_cols = [f"att_{t}" for t in teams]
    def_cols = [f"def_{t}" for t in teams]
    # home features: att_home + def_away
    row_home = pd.Series(0, index=att_cols + def_cols)
    row_home[f"att_{home_team}"] = 1
    row_home[f"def_{away_team}"] = 1

    # away features: att_away + def_home
    row_away = pd.Series(0, index=att_cols + def_cols)
    row_away[f"att_{away_team}"] = 1
    row_away[f"def_{home_team}"] = 1
    return row_home.to_frame().T, row_away.to_frame().T


def predict_lambdas(model_bundle_path: str, fixtures_df: pd.DataFrame):
    """
    给定未来比赛（fixtures_df 含 home_team, away_team, date 可选），返回 lambda_h 与 lambda_a
    fixtures_df 必须包含 home_team 与 away_team 列
    返回 DataFrame with lambda_h, lambda_a
    """
    bundle = joblib.load(model_bundle_path)
    teams = bundle['teams']
    m_home = bundle['model_home']
    m_away = bundle['model_away']

    rows = []
    for _, r in fixtures_df.iterrows():
        home, away = r['home_team'], r['away_team']
        if home not in teams or away not in teams:
            # Unknown team: fallback to global mean
            lambda_h = np.clip(m_home.predict(np.zeros((1, len(bundle['feature_home_cols']))))[0], 0.01, None)
            lambda_a = np.clip(m_away.predict(np.zeros((1, len(bundle['feature_away_cols']))))[0], 0.01, None)
        else:
            row_home, row_away = _build_feature_row(home, away, teams)
            lambda_h = np.clip(m_home.predict(row_home[bundle['feature_home_cols']].values)[0], 0.01, None)
            lambda_a = np.clip(m_away.predict(row_away[bundle['feature_away_cols']].values)[0], 0.01, None)
        rows.append({'date': r.get('date', None), 'home_team': home, 'away_team': away, 'lambda_h': lambda_h, 'lambda_a': lambda_a})
    return pd.DataFrame(rows)


def bivariate_poisson_joint_pmf(lambda_h, lambda_a, lambda_0, max_goals=6):
    """
    通过 Poisson 共享成分分解计算联合概率：
    设 U ~ Pois(lambda_0) 共有成分，V1 ~ Pois(lambda_h - lambda_0)，V2 ~ Pois(lambda_a - lambda_0)
    则 Home = U + V1, Away = U + V2
    要求 0 <= lambda_0 <= min(lambda_h, lambda_a)
    返回矩阵 P[i,j] = P(Home=i, Away=j) for i,j=0..max_goals
    """
    # 检查
    lambda_h = float(lambda_h); lambda_a = float(lambda_a); lambda_0 = float(lambda_0)
    if lambda_0 < 0:
        raise ValueError("lambda_0 must be >= 0")
    lambda_0 = min(lambda_0, lambda_h, lambda_a)
    lam1 = max(lambda_h - lambda_0, 0.0)
    lam2 = max(lambda_a - lambda_0, 0.0)

    # Precompute pmfs
    max_k = max_goals
    pU = [poisson.pmf(k, lambda_0) for k in range(max_k+1)]
    pV1 = [poisson.pmf(k, lam1) for k in range(max_k+1)]
    pV2 = [poisson.pmf(k, lam2) for k in range(max_k+1)]

    P = np.zeros((max_k+1, max_k+1))
    # Sum over shared component u
    for u in range(0, max_k+1):
        for v1 in range(0, max_k+1 - u):
            for v2 in range(0, max_k+1 - u):
                i = u + v1
                j = u + v2
                if i <= max_k and j <= max_k:
                    P[i, j] += pU[u] * pV1[v1] * pV2[v2]
    return P


def outcome_probabilities_from_lambdas(lambda_h, lambda_a, rho=0.1, max_goals=6):
    """
    给定两个边缘 lambda_h, lambda_a，使用共享强度 lambda_0 = rho * min(lambda_h, lambda_a)
    计算 P(H), P(D), P(A)
    返回 dict
    """
    lambda_0 = rho * min(lambda_h, lambda_a)
    P = bivariate_poisson_joint_pmf(lambda_h, lambda_a, lambda_0, max_goals=max_goals)
    p_home_win = float(np.triu(P, k=1).sum())
    p_draw = float(np.trace(P))
    p_away_win = float(np.tril(P, k=-1).sum())
    # normalize in case tail mass beyond max_goals
    s = p_home_win + p_draw + p_away_win
    if s <= 0:
        return {'p_H': 0.33, 'p_D': 0.34, 'p_A': 0.33}
    return {'p_H': p_home_win / s, 'p_D': p_draw / s, 'p_A': p_away_win / s}


def predict_probs_for_fixtures(model_bundle_path: str, fixtures_df: pd.DataFrame, rho: float = 0.08, max_goals: int = 6):
    lambdas = predict_lambdas(model_bundle_path, fixtures_df)
    rows = []
    for _, r in lambdas.iterrows():
        lp = outcome_probabilities_from_lambdas(r['lambda_h'], r['lambda_a'], rho=rho, max_goals=max_goals)
        rows.append({**r.to_dict(), **lp})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 简单演示：使用 data/raw/example_matches.csv 训练并预测同样的比赛（仅示例）
    matches = os.path.join(ROOT, 'data', 'raw', 'example_matches.csv')
    model_path = fit_poisson_pair(matches)
    fixtures = pd.read_csv(matches, parse_dates=['date']).head(10)[['date','home_team','away_team']]
    res = predict_probs_for_fixtures(model_path, fixtures, rho=0.08)
    print(res.head())
