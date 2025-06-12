from scripts.train.models.xgboost_model import train_xgboost

data_path = "data/processed/train_selected_cleaned.csv"
train_xgboost(data_path)