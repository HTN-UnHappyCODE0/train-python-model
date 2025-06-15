# scripts/train/train_logistic_model.py
import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

def train_logistic_model(data_path):
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

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - start

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    log_path = f"logs/train_logistic_report.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Logistic Regression Training - {timestamp} ===\n")
        f.write(f"⏱️ Training time: {duration:.2f}s\n")
        f.write(f"🎯 Validation Accuracy: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred, labels=y.unique())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix_logistic_{timestamp}.png")
    plt.close()

    joblib.dump(pipeline, f"models/logistic_model_{timestamp}.pkl")

if __name__ == "__main__":
    train_logistic_model("data/ObesityDataSet_raw_and_data_sinthetic.csv")
