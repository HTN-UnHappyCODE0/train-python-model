import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

def train_logistic_regression(data_path):
    df = pd.read_csv(data_path)

    # Kiểm tra nhãn
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    # Tách dữ liệu
    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Tạo thư mục lưu output
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Lưu báo cáo đánh giá
    with open("logs/logistic_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig("figures/logistic_confusion_matrix.png")
    plt.close()

    # SHAP summary plot
    explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)  # show=False để không hiện popup
    plt.title("SHAP Summary - Logistic Regression")
    plt.savefig("figures/logistic_shap_summary.png", bbox_inches='tight')
    plt.close()
