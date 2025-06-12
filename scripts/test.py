import pandas as pd

df = pd.read_csv("data/processed/train_selected_cleaned.csv")
print(df.columns.tolist())
