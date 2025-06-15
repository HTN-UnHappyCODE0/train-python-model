import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_random_forest(data_path):
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Tiền xử lý
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Fit riêng để lấy tên feature
    preprocessor.fit(X_train)

    if categorical_features:
        ohe_fitted = preprocessor.named_transformers_['cat']
        cat_feature_names = ohe_fitted.get_feature_names_out(categorical_features)
        feature_names = numeric_features + cat_feature_names.tolist()
    else:
        feature_names = numeric_features

    # Pipeline model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # ⏱️ Đo thời gian huấn luyện
    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_time
    print(f"⏱️ Thời gian huấn luyện: {train_duration:.2f} giây")

    y_pred = model_pipeline.predict(X_test)

    # 🎯 Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Độ chính xác: {accuracy:.4f}")

    # Ghi báo cáo
    os.makedirs("logs", exist_ok=True)
    with open("logs/random_forest_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    # 📊 Vẽ Confusion Matrix
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred, labels=y.unique())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("figures/random_forest_confusion_matrix.png")
    plt.close()

    # SHAP
    model = model_pipeline.named_steps['classifier']
    X_test_transformed = preprocessor.transform(X_test)
    X_test_shap = pd.DataFrame(
        X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
        columns=feature_names
    )

    explainer = shap.Explainer(model, X_test_transformed)
    shap_values = explainer(X_test_transformed, check_additivity=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_shap, show=False)
    plt.title("SHAP Summary - Random Forest")
    plt.savefig("figures/random_forest_shap_summary.png", bbox_inches="tight")
    plt.close()

    # 💾 Lưu model với timestamp
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/random_forest_model_{timestamp}.pkl"
    joblib.dump(model_pipeline, model_path)
    print(f"✅ Mô hình đã được lưu tại: {model_path}")
