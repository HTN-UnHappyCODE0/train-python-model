from scripts.train.models.train_random_forest import train_random_forest

data_path = "data/processed/train_selected_cleaned.csv"
train_random_forest(data_path)