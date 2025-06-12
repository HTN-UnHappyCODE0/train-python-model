import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_random_forest(data_path):
    # Đọc dữ liệu
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Tách cột số và cột phân loại
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Tách tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tiền xử lý: scale số & encode phân loại
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    # Pipeline: Tiền xử lý + RandomForest
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Huấn luyện
    model_pipeline.fit(X_train, y_train)

    # Dự đoán
    y_pred = model_pipeline.predict(X_test)

    # Lưu báo cáo
    os.makedirs("logs", exist_ok=True)
    with open("logs/random_forest_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    # Ma trận nhầm lẫn
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("figures/random_forest_confusion_matrix.png")
    plt.close()

    # Tính SHAP values
    # Trích mô hình và dữ liệu đã xử lý từ pipeline
    model = model_pipeline.named_steps['classifier']
    X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)

    explainer = shap.Explainer(model, X_test_transformed)
    shap_values = explainer(X_test_transformed, check_additivity=False)

    # Tên các feature sau khi encode (dành cho SHAP plot)
    ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = numeric_features + cat_feature_names.tolist()

    X_test_shap = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
                               columns=feature_names)

    # SHAP summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_shap, show=False)
    plt.title("SHAP Summary - Random Forest")
    plt.savefig("figures/random_forest_shap_summary.png", bbox_inches="tight")
    plt.close()
