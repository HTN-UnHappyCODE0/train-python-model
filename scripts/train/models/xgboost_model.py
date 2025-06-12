import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

def train_xgboost(data_path):
    df = pd.read_csv(data_path)

    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    os.makedirs("logs", exist_ok=True)
    with open("logs/xgboost_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Oranges")
    plt.title("XGBoost Confusion Matrix")
    plt.savefig("figures/xgboost_confusion_matrix.png")
    plt.close()

    # SHAP summary plot
    explainer = shap.Explainer(model, X_test_scaled)
    shap_values = explainer(X_test_scaled)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary - XGBoost")
    plt.savefig("figures/xgboost_shap_summary.png", bbox_inches="tight")
    plt.close()
