# football-predictor

完整的中文说明（模板项目）：从比赛数据抓取 → 特征工程 → LightGBM 多分类建模 → 按赔率回测的研究性项目。

> 注意：本项目仅作研究和教学用途，不作为实盘或博彩建议。在使用真实资金前，请做好充分回测与风控，并遵守当地法律法规与平台规则。

## 仓库说明（中文）
本仓库提供一个最小可运行的足彩/足球比赛预测研究模板，包含：

- 数据抓取示例（FBref 页面抓取、football-data.org API 示例）
- 特征工程：按球队滚动窗口统计、简单 Elo、目标编码
- 模型训练：LightGBM 多分类（主胜/平/客胜）示例并含时间序列 CV
- 预测与回测：按赔率计算边值（edge），按策略回测盈亏
- 辅助工具：合并赔率、数据对齐等工具函数

### 项目结构
- data/
  - raw/            （放原始抓取数据 CSV）
  - processed/      （特征工程后的数据与预测结果）
- src/
  - fetch_fbref.py                  # 从 FBref 抓取（示例）
  - fetch_footballdata_api.py       # football-data.org API 抓取（示例）
  - features.py                     # 特征工程脚本
  - model_train.py                  # 训练 LightGBM 并保存模型
  - predict.py                      # 使用训练好的模型进行预测
  - backtest.py                     # 回测脚本（基于赔率与模型概率）
  - utils.py                        # 合并赔率等辅助函数
- models/                            # 训练好的模型会保存到这里
- requirements.txt                   # 依赖清单
- README.md                          # 本文件（中文说明）
- .gitignore                         # Python 常见忽略文件
- LICENSE (MIT)                      # 开源许可

> 当前仓库可能只显示了初始的 `.gitignore` 与 `LICENSE`，我刚刚已把本 README 添加到仓库中；你刷新页面后应该能看到完整 README。之所以最初只有两个文件，是因为仓库创建时我们选择了仅初始化 .gitignore 与 License，后续再逐步添加项目文件。

## 快速开始（在本仓库已有完整代码后）
以下命令假设你已把仓库克隆到本地或通过 GitHub 下载 ZIP 并解压。

1. 创建并激活虚拟环境（Linux / macOS）

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell)：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. （可选）配置 football-data API token
- 如果要使用 `src/fetch_footballdata_api.py`，请申请 token（https://www.football-data.org）并在项目根目录创建 `.env`：

```
FOOTBALL_DATA_API_TOKEN=your_token_here
```

4. 抓取数据（任选一种方法）

- FBref 抓取（示例，可能需根据联赛调整 `league_slug` 与 `season`）：

```bash
python src/fetch_fbref.py
```

- football-data.org（需要 token）：

```bash
python src/fetch_footballdata_api.py
```

抓取后原始 CSV 会保存在 `data/raw/` 下。

5. 生成特征

```bash
python -c "from src.features import build_features; build_features('data/raw/your_matches.csv','data/processed/processed_matches.csv')"
```

> 注意替换 `'data/raw/your_matches.csv'` 为你实际抓取的文件名。

6. 训练模型

```bash
python src/model_train.py
```

训练后模型保存在 `models/lgb_multi_model.pkl`。

7. 预测（示例）

准备一个包含与训练时相同特征列的待预测 CSV（例如把部分比赛当作“未来”），然后：

```bash
python src/predict.py --input data/processed/processed_matches_future.csv --model models/lgb_multi_model.pkl
```

输出会写到 `data/processed/predictions.csv`，其中包含 p_H/p_D/p_A（模型概率）。

8. 合并赔率（回测必须）

回测脚本需要 `odd_h`/`odd_d`/`odd_a` 列（对应主胜/平/客胜下注赔率）。你需要准备历史赔率 CSV（列：date, home_team, away_team, odd_h, odd_d, odd_a），然后用 `src/utils.py` 中的 `merge_odds` 方法合并：

```python
from src.utils import merge_odds
import pandas as pd
matches = pd.read_csv('data/processed/processed_matches.csv', parse_dates=['date'])
merged = merge_odds(matches, 'data/raw/odds_sample.csv')
merged.to_csv('data/processed/processed_matches_with_odds.csv', index=False)
```

9. 回测

```bash
python src/backtest.py
```

回测会读取 `data/processed/predictions.csv` 并输出累计收益图 `backtest_balance.png` 与控制台统计。

## 数据与注意事项（重要）
- 球队名称规范化：不同数据源（FBref、football-data、赔率站）队名可能不一致。建议准备一份名称映射表（mapping）在合并前统一。否则合并失败或匹配错误会导致回测不准确。
- 赔率数据：历史赔率常常需要付费或自己爬取。请注意目标站点的使用规则与法律合规性。
- 避免未来泄露（data leakage）：训练/验证/测试一定要按时间分割，代码中使用了 TimeSeriesSplit 做演示。
- 模型与特征：模板只是入门示例。可以尝试：双泊松 / bivariate Poisson、基于进球回归的概率计算、更丰富的事件特征（若使用 StatsBomb 等事件数据）。

## 若要下载整个项目 ZIP
在仓库页面点击绿色 "Code" 按钮，然后选择 "Download ZIP"，或直接访问：

```
https://github.com/nidaye286077907-cpu/football-predictor/archive/refs/heads/main.zip
```

（刷新页面后 README 会同步显示）

## 下一步我可以为你做的事（选项）
1. 我可以把示例数据（小量真实历史赛果和示例赔率 CSV）上传到 `data/raw/example_matches.csv` 与 `data/raw/example_odds.csv`，便于你直接跑通全流程；
2. 我可以把 Poisson / bivariate Poisson 的概率转换代码补入 `src/poisson.py`，并提供如何用回归预测进球数再合成胜平负概率的示例；
3. 生成项目的 ZIP 下载链接并把示例数据打包（我也可以直接在仓库里添加示例数据文件）。

请回复告诉我你要我接下来做哪一项（例如：1 或 2 或 3），我会立即继续并把变更提交到仓库并把 ZIP 链接给你。