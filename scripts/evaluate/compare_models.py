import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.multiclass import unique_labels
from datetime import datetime

def evaluate_model(model_name, model_path, X_val, y_val, labels=None, save_dir="evaluation_results"):
    model = joblib.load(model_path)
    y_pred = model.predict(X_val)

    # Lấy labels nếu chưa truyền vào
    if labels is None:
        labels = unique_labels(y_val, y_pred)
    target_names = [str(label) for label in labels]

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    # Report
    report = classification_report(y_val, y_pred, target_names=target_names)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(save_dir, f"{model_name}_report_{timestamp}.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== {model_name} Evaluation ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix Heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix_{timestamp}.png"))
    plt.close()

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
