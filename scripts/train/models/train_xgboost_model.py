import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from xgboost import XGBClassifier

def train_xgboost_model(data_path):
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    preprocessor.fit(X_train)

    if categorical_features:
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + cat_names.tolist()
    else:
        feature_names = numeric_features

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])

    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_time
    print(f"⏱️ Thời gian huấn luyện: {train_duration:.2f} giây")

    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Độ chính xác: {accuracy:.4f}")

    # 📁 Tạo thư mục
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 📝 Ghi log
    log_path = f"logs/train_xgboost_model_report.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Run at {timestamp} ===\n")
        f.write(f"⏱️ Training Time: {train_duration:.2f}s\n")
        f.write(f"🎯 Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    # 📊 Vẽ Confusion Matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred, labels=y.unique())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"figures/xgboost_confusion_matrix_{timestamp}.png", bbox_inches="tight")
    plt.close()

    # 🔍 SHAP
    model = model_pipeline.named_steps['classifier']
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_test_transformed, "toarray"):
        X_test_shap = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)
    else:
        X_test_shap = pd.DataFrame(X_test_transformed, columns=feature_names)

    explainer = shap.Explainer(model, X_test_transformed)
    shap_values = explainer(X_test_transformed, check_additivity=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_shap, show=False)
    plt.title("SHAP Summary - XGBoost")
    plt.savefig(f"figures/xgboost_shap_summary_{timestamp}.png", bbox_inches="tight")
    plt.close()
