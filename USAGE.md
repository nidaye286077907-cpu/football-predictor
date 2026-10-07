# 使用手册（中文）

这是 football-predictor 项目的完整使用手册，覆盖从环境准备、数据抓取、特征工程、模型训练、Poisson 模型与 rho 网格搜索、预测与回测的所有步骤。

## 目录
1. 环境准备
2. 数据说明
3. 快速演示（使用示例数据）
4. 使用 football-data API
5. 特征工程说明
6. LightGBM 模型训练
7. Poisson 模型与 rho 网格搜索
8. 回测说明
9. 部署与进阶建议

---

## 1. 环境准备
- 推荐 Python 3.8+（建议 3.10）
- 克隆仓库并创建虚拟环境：

```bash
git clone https://github.com/nidaye286077907-cpu/football-predictor.git
cd football-predictor
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 数据说明
- data/raw/example_matches.csv: 示例赛果数据，包含 date, home_team, away_team, home_goals, away_goals
- data/raw/example_odds.csv: 示例赔率，包含 date, home_team, away_team, odd_h, odd_d, odd_a

如果你需要真实数据：
- 可用 FBref 抓取（示例脚本 src/fetch_fbref.py）
- 可用 football-data.org API（示例脚本 src/fetch_footballdata_api.py），需在 .env 中配置 TOKEN
- 高级事件数据：StatsBomb（需注册并下载 open-data）

## 3. 快速演示（使用示例数据）
1. 生成特征并训练 LightGBM：

```bash
python -c "from src.features import build_features; build_features('data/raw/example_matches.csv','data/processed/processed_matches.csv')"
python src/model_train.py
```

2. 使用 LightGBM 预测并回测（示例）：

```bash
python src/predict.py --input data/processed/processed_matches.csv --model models/lgb_multi_model.pkl
python src/backtest.py
```

3. 使用 Poisson 模型并做 rho 网格搜索（示例）：

```bash
# 训练并在 test 划分上做 rho 网格搜索，结果保存到 data/processed/rho_grid_results.csv
python -c "from src.rho_grid_search import run_rho_grid_search; run_rho_grid_search()"
```

4. 查看结果：
- 网格搜索结果：data/processed/rho_grid_results.csv
- 最优预测：data/processed/predictions_best_rho.csv
- 各 rho 的回测图：data/processed/backtest_balance_rho_*.png

## 4. 使用 football-data API
1. 在 https://www.football-data.org/ 注册并获取 API Token
2. 复制 .env.example 为 .env 并填入 TOKEN：

```
FOOTBALL_DATA_API_TOKEN=your_token_here
```

3. 运行示例抓取：

```bash
python src/fetch_footballdata_api.py
```

注意：免费套餐有请求频率和部分端点限制。商业/大规模使用请升级套餐或联系服务方。

## 5. 特征工程说明
特征脚本 src/features.py 会：
- 将原始比赛记录按照日期排序
- 计算每支球队近 N 场（默认 5 场）的平均进球/失球与胜负形态
- 构建主队/客队各自的滚动特征并合并为比赛级特征
- 生成简单 Elo 指标（用于衡量队力）

输出示例文件：data/processed/processed_matches.csv，包含模型所需特征列。

## 6. LightGBM 模型训练
- 模型文件：src/model_train.py
- 使用 TimeSeriesSplit 做时间序列交叉验证，避免未来泄露
- 输出模型保存在 models/lgb_multi_model.pkl

## 7. Poisson 模型与 rho 网格搜索
- src/poisson.py: 训练两个 Poisson 回归（主队与客队进球），并使用共享成分 rho 构造二元泊松联合概率
- src/rho_grid_search.py: 在训练/测试切分上对 rho 做网格搜索：
  - 对每个 rho 生成预测、合并赔率、运行回测
  - 保存每个 rho 的最终余额、ROI 与下注次数到 data/processed/rho_grid_results.csv
  - 将最优 rho 的预测保存为 data/processed/predictions_best_rho.csv，并保存回测图

建议流程：
1. 准备历史数据并按时间排序
2. 确认赔率数据覆盖测试区间并规范球队名称
3. 使用 run_rho_grid_search 进行网格搜索并查看结果

## 8. 回测说明
- 回测脚本：src/backtest.py
- 回测逻辑（简单版）：每场比赛选择 edge 最大的盘口（模型概率 - 赔率隐含概率），若 edge >= min_edge 则按固定比例下注
- 配置参数：bankroll、stake_frac（每笔下注占初始本金比例）、min_edge

注意：该回测为示例，未实现滑点、限额、投注上限、并发赛程的资金限制等真实交易因素。实盘前请加入更严格风险控制逻辑（止损、最大回撤、仓位限制）。

## 9. 部署与进阶建议
- 若需实时预测，建议部署为定时任务或 API 服务，频率不要超免费 API 的配额
- 对比 Poisson 与 ML 模型：Poisson 更具解释性，ML（LightGBM）能利用更多特征
- 提高模型：加入 xG 特征、伤停、赛程密度、换帅等外部信息
- 优化回测：考虑本金动态调整、凯利分配、不同赔率来源的套利

---

如果你需要，我可以继续帮你：
- 把 Notebook 增加交互控件（ipywidgets）以便在浏览器中实时调整 rho、stake_frac 等参数；
- 添加自动化的 rho 优化（例如使用 CV 折叠或按联赛分别优化）；
- 集成更真实的赔率数据源（示例：OddsPortal 爬虫，或第三方 API）。
