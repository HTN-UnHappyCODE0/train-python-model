from scripts.train.models.logistic_regression import train_logistic_regression
from scripts.train.models.random_forest import train_random_forest
from scripts.train.models.xgboost_model import train_xgboost


path = "data/processed/train_selected_cleaned.csv"
train_logistic_regression(path)
train_random_forest(path)
train_xgboost(path)