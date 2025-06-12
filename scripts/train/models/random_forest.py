import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

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

    # Scale dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Lưu báo cáo phân loại
    os.makedirs("logs", exist_ok=True)
    with open("logs/random_forest_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    # Vẽ ma trận nhầm lẫn
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("figures/random_forest_confusion_matrix.png")
    plt.close()

    # Tính SHAP values và vẽ biểu đồ
    explainer = shap.Explainer(model, X_test_scaled)
    shap_values = explainer(X_test_scaled,check_additivity=False)

    # Vẽ SHAP summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary - Random Forest")
    plt.savefig("figures/random_forest_shap_summary.png", bbox_inches="tight")
    plt.close()
