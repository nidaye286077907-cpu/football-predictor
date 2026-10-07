新增 CLI 工具：

使用方式示例：

# 生成特征
python src/cli.py features --raw data/raw/example_matches.csv --out data/processed/processed_matches.csv

# 训练 LightGBM
python src/cli.py train --processed data/processed/processed_matches.csv

# 训练 Poisson
python src/cli.py train-poisson --matches data/raw/example_matches.csv

# 用 LightGBM 预测
python src/cli.py predict --model lightgbm --input data/processed/processed_matches.csv

# 用 Poisson 预测
python src/cli.py predict --model poisson --input data/processed/processed_matches.csv --model-path models/poisson_pair_models.pkl --rho 0.08

# 回测
python src/cli.py backtest --predictions data/processed/predictions.csv

# rho 网格搜索
python src/cli.py rho-grid

