from scripts.train.models.tune_random_forest import tune_random_forest

data_path = "data/processed/train_selected_cleaned.csv"
tune_random_forest(data_path)
