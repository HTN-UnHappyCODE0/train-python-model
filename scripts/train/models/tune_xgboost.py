import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import joblib
from datetime import datetime

def tune_xgboost(data_path):
    # 📦 Load dữ liệu
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Thiếu cột 'NObeyesdad' trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # ✂️ Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # 🧪 Không gian tham số
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.3],
        'classifier__reg_lambda': [0.5, 1.0, 1.5],
        'classifier__reg_alpha': [0, 0.1, 0.5],
    }

    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        n_jobs=-1,
        scoring="accuracy",
        return_train_score=True
    )

    print("🚀 Bắt đầu tối ưu hóa tham số cho XGBoost...")
    start = time.time()
    search.fit(X_train, y_train)
    duration = time.time() - start
    print("✅ Tối ưu hóa hoàn tất!")

    best_pipeline = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    # 📝 Tạo thư mục
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 🕒 Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 📋 Ghi log
    log_path = f"logs/tune_xgboost_report.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Run at {timestamp} ===\n")
        f.write(f"⏱️ Duration: {duration:.2f} seconds\n")
        f.write(f"🎯 Best Accuracy (CV): {best_score:.4f}\n")
        f.write("🔧 Best Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  - {k}: {v}\n")

        # 🔍 Ghi lại tất cả các lần thử
        f.write("\n📊 Tất cả các lần thử:\n")
        results = pd.DataFrame(search.cv_results_)
        cols_to_log = ['mean_test_score', 'std_test_score', 'rank_test_score'] + [
            k for k in results.columns if k.startswith('param_')
        ]
        results_to_log = results[cols_to_log].sort_values(by="rank_test_score")
        for idx, row in results_to_log.iterrows():
            f.write(f"\n  🔁 Thử {idx + 1}:\n")
            for col in cols_to_log:
                f.write(f"    {col}: {row[col]}\n")

        # 🔄 Lưu bảng kết quả ra CSV
        csv_path = f"logs/tune_xgboost_all_trials_{timestamp}.csv"
        results_to_log.to_csv(csv_path, index=False)

    # 📈 Đánh giá trên tập validation
    y_pred_val = best_pipeline.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"🎯 Accuracy trên tập validation: {acc_val:.4f}")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n🎯 Validation Accuracy: {acc_val:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_val, y_pred_val))

    # 🔄 Lưu pipeline model
    model_path = f"models/xgboost_best_{timestamp}.pkl"
    joblib.dump(best_pipeline, model_path)

    # 📊 Vẽ confusion matrix
    cm = confusion_matrix(y_val, y_pred_val, labels=y.unique())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix_xgboost_{timestamp}.png")
    plt.close()

    print(f"📦 Model saved to: {model_path}")
    print(f"📝 Log saved to: {log_path}")
    print(f"📄 CSV trials saved to: {csv_path}")
