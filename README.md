# football-predictor

这是一个完整的足球比赛预测研究模板，适合学习和实验：
- 数据抓取：FBref / football-data API 示例
- 特征工程：近期表现、主客场、进失球率、让球/赔率特征
- 模型训练：LightGBM 多分类（主胜 / 平 / 客胜）
- 预测：加载模型，输出胜平负概率
- 回测：基于历史赔率做盈亏仿真

> 仅用于研究和学习；不构成任何投注建议或实盘推荐。

## 目录结构

```text
football-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .env.example
├── data/
│   ├── raw/
│   │   ├── example_matches.csv
│   │   └── example_odds.csv
│   └── processed/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── fetch_fbref.py
│   ├── fetch_footballdata_api.py
│   ├── features.py
│   ├── model_train.py
│   ├── predict.py
│   ├── backtest.py
│   └── utils.py
├── models/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```

## 特性

- 生成滚动窗口特征：近 5 场球队表现、进球/失球、胜率、主客场表现
- 简单 Elo 特征：主队和客队预赛 Elo 差值
- 基于 `LightGBM` 多分类模型输出 `P(H) / P(D) / P(A)`
- 回测脚本：比较模型概率与赔率，计算边值（edge）和盈亏

## 环境准备

### 1) Python 环境
建议使用 Python 3.10+，创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

## 运行流程

### 方式 A：使用示例数据（最简单）
这个仓库自带示例数据，足以跑通完整流程：

```bash
python -c "from src.features import build_features; build_features('data/raw/example_matches.csv', 'data/processed/processed_matches.csv')"
python src/model_train.py
python src/predict.py --input data/processed/processed_matches.csv --model models/lgb_multi_model.pkl
python src/backtest.py
```

### 方式 B：抓取真实数据

#### FBref（示例）
需要根据联赛和赛季调整 `league_slug` 和 `season`：

```bash
python src/fetch_fbref.py
```

#### football-data API（推荐）
申请 token 后在项目根目录创建 `.env`：

```env
FOOTBALL_DATA_API_TOKEN=your_token_here
```

然后运行：

```bash
python src/fetch_footballdata_api.py
```

## 实际建模步骤

### 1. 抓取比赛数据
可以从以下来源获取：
- FBref：网页/表格抓取
- football-data.org：官方 API
- OpenFootball：开放数据集
- StatsBomb Open Data：事件数据（适合高级分析）

### 2. 清洗和统一字段
关键字段：
- `date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`

### 3. 特征工程
`src/features.py` 已经实现了：
- 近 5 场主队/客队进球与失球
- 胜/平/负滚动结果
- 主客场近期表现差异
- 简单 Elo 差值

### 4. 模型训练
`src/model_train.py` 使用 LightGBM 多分类：
- `0` = 主胜（H）
- `1` = 平（D）
- `2` = 客胜（A）

### 5. 预测与回测
`src/predict.py` 输出模型概率：
- `p_H`
- `p_D`
- `p_A`

`src/backtest.py` 会读取赔率并比较：
- 模型给出的概率
- 市场赔率隐含概率
- 选取最大 edge 的投注选项
- 输出累计盈亏图与 ROI

## 使用示例赔率文件
回测需要 `odd_h` / `odd_d` / `odd_a`。你可以使用 `data/raw/example_odds.csv` 作为示例文件：

```bash
python -c "from src.utils import merge_odds; import pandas as pd; m = pd.read_csv('data/raw/example_matches.csv', parse_dates=['date']); print(merge_odds(m, 'data/raw/example_odds.csv').head())"
```

## 常见问题

### 1) 队名不一致
跨数据源时，队名可能不同：
- `Manchester United` vs `Man United`
- `West Ham` vs `West Ham United`

处理方法：在合并前建立一张队名映射表。`

### 2) 没有真实赔率
很多免费数据源不提供完整的历史赔率。你可以：
- 使用第三方 API
- 自己爬取网站数据
- 做离线回测（按历史比赛结果和模拟赔率）

### 3) 数据泄露风险
训练/验证/测试必须按时间顺序切分，不要用未来信息去训练当前。这个模板已通过 `TimeSeriesSplit` 作为示例。

## 进阶方向

- 双泊松 / bivariate Poisson：更贴近进球数分布
- xG（期望进球）模型：基于事件数据做高质量特征
- XGBoost / CatBoost：可替代 LightGBM
- 实时赔率与市场效率研究：对模型和赔率差值做更深入分析

## 免责声明

本项目不提供保证收益，也不构成赌博、下注或博彩建议。请遵守当地法律和平台规则，并在使用真实资金前进行严格回测、风险控制和监测。

## 开源协议

MIT License

---

如果你要，我还可以继续为你补：
1. 完整的 `Poisson` 版本（更经典的胜平负预测方法）
2. `StatsBomb` / `kloppy` 的高级事件数据版本
3. `Jupyter Notebook` 版本，可直接在浏览器中运行和可视化
4. 一个更真实的 `example_matches.csv` / `example_odds.csv` 示例，覆盖更多赛季和球队

你只要告诉我你想继续哪一个，我就继续补代码和示例数据。































































































































































































































































































s





























































































































































































































































































































































































































































































































n









































































n
























































































