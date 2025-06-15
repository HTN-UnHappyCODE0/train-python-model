# scripts/train/run_tune_logistic.py
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def tune_logistic_model(data_path):
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Thiếu cột 'NObeyesdad' trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    logistic = LogisticRegression(max_iter=1000, random_state=42)

    param_dist = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
        "class_weight": [None, "balanced"],
        "max_iter": [1000, 2000, 3000]
    }

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", logistic)
    ])

    search = RandomizedSearchCV(
        pipe, param_distributions={ "classifier__" + k: v for k, v in param_dist.items() },
        n_iter=10,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        random_state=42
    )

    print("🚀 Bắt đầu tuning Logistic Regression...")
    start = time.time()
    search.fit(X_train, y_train)
    duration = time.time() - start
    print("✅ Tuning hoàn tất!")

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 🔍 Ghi log chi tiết từng lần
    log_path = f"logs/tune_logistic_report.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Logistic Tuning Run - {timestamp} ===\n")
        f.write(f"⏱️ Duration: {duration:.2f}s\n")
        f.write(f"🎯 Best CV Accuracy: {best_score:.4f}\n")
        f.write("🔧 Best Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  - {k}: {v}\n")

        # Ghi lại toàn bộ lịch sử thử nghiệm
        f.write("\n📊 All Trials:\n")
        results = pd.DataFrame(search.cv_results_)
        for i in range(len(results)):
            f.write(f"\nTrial {i+1} - Mean CV Accuracy: {results['mean_test_score'][i]:.4f}\n")
            params = results['params'][i]
            for k, v in params.items():
                f.write(f"  - {k}: {v}\n")

    # 🧪 Validation
    y_pred_val = best_model.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n🎯 Validation Accuracy: {acc_val:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_val, y_pred_val))

    cm = confusion_matrix(y_val, y_pred_val, labels=y.unique())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title("Confusion Matrix - Logistic Regression (Tuned)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix_logistic_tuned_{timestamp}.png")
    plt.close()

    joblib.dump(best_model, f"models/logistic_best_{timestamp}.pkl")

if __name__ == "__main__":
    tune_logistic_model("data/ObesityDataSet_raw_and_data_sinthetic.csv")
