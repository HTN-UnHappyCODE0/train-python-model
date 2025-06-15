from scripts.train.models.train_xgboost_model import train_xgboost_model

data_path = "data/processed/train_selected_cleaned.csv"
train_xgboost_model(data_path)