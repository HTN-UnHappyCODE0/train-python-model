import os
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def tune_random_forest(data_path):
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Chia dữ liệu train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    rf = RandomForestClassifier(random_state=42)

    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__class_weight': [None, 'balanced']
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("⏳ Bắt đầu tối ưu hóa tham số...")
    search.fit(X_train, y_train)
    print("✅ Tối ưu hoàn tất!")

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    # Dự đoán và đánh giá trên tập validation
    y_val_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)

    # Tạo thư mục lưu
    os.makedirs("tunes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Lưu mô hình
    dump(best_model, "tunes/random_forest_best.pkl")

    # Tạo log file theo timestamp
    log_path = "logs/tune_random_forest_report.txt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(log_path, "a",encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"🕒 Run Time: {timestamp}\n")
        f.write("=== TUNING RANDOM FOREST ===\n")
        f.write("\nBest Parameters:\n")
        f.write(str(best_params))
        f.write("\n\nBest Cross-Validation Score:\n")
        f.write(str(best_score))
        f.write(f"\n\nAccuracy on Validation Set: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_val, y_val_pred))

    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.savefig(f"figures/confusion_matrix_rf_{timestamp}.png", bbox_inches="tight")
    plt.close()
