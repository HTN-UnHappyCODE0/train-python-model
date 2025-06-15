from scripts.evaluate.compare_models import evaluate_model
import joblib
import pandas as pd

# Load dữ liệu validation
X_val = joblib.load("data/X_val.pkl")
y_val = joblib.load("data/y_val.pkl")
labels = y_val.unique().tolist()

# Danh sách model
models = [
    ("LogisticRegression", "models/logistic_best_2025-06-15_10-30-44.pkl"),
    ("RandomForest", "models/random_forest_best_2025-06-15_12-43-17.pkl"),
    ("XGBoost", "models/xgboost_best_2025-06-15_12-47-54.pkl"),
]

# Chạy đánh giá từng mô hình
results = []
for name, path in models:
    r = evaluate_model(name, path, X_val, y_val, labels)
    results.append(r)

# Xuất ra bảng tổng hợp
df = pd.DataFrame(results)
df.to_csv("evaluation_results/model_comparison.csv", index=False)
print(df.sort_values("f1_score", ascending=False))
