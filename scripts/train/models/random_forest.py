import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_random_forest(data_path):
    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Chia dữ liệu train/test
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

    # Fit riêng preprocessor để lấy feature names
    preprocessor.fit(X_train)

    # Lấy tên feature sau khi one-hot encode
    ohe_fitted = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe_fitted.get_feature_names_out(categorical_features)
    feature_names = numeric_features + cat_feature_names.tolist()

    # Tạo pipeline với model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    # Tạo thư mục kết quả
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Ghi báo cáo
    with open("logs/random_forest_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    # Ma trận nhầm lẫn
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("figures/random_forest_confusion_matrix.png")
    plt.close()

    # SHAP
    model = model_pipeline.named_steps['classifier']
    X_test_transformed = preprocessor.transform(X_test)

    explainer = shap.Explainer(model, X_test_transformed)
    shap_values = explainer(X_test_transformed, check_additivity=False)

    # X_test đã transform thành numpy array, tạo lại DataFrame với tên cột
    X_test_shap = pd.DataFrame(
        X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
        columns=feature_names
    )

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_shap, show=False)
    plt.title("SHAP Summary - Random Forest")
    plt.savefig("figures/random_forest_shap_summary.png", bbox_inches="tight")
    plt.close()
