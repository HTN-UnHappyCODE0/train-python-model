# scripts/train/run_tune_xgboost.py

from scripts.train.models.tune_xgboost import tune_xgboost

data_path = "data/processed/train_selected_cleaned.csv"
tune_xgboost(data_path)
