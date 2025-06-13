import pandas as pd
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

RANDOM_STATE = 42
N_ITER_RANDOM_SEARCH_RF = 20
CV_FOLDS_OPTIMIZATION = 5

def tune_random_forest(data_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df = pd.read_csv(data_path)
    if 'NObeyesdad' not in df.columns:
        raise ValueError("Cột 'NObeyesdad' không tồn tại trong dữ liệu.")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    # Tách tập validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Tối ưu hóa tham số
    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    random_search_rf = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist_rf,
        n_iter=N_ITER_RANDOM_SEARCH_RF,
        cv=StratifiedKFold(n_splits=CV_FOLDS_OPTIMIZATION, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print("⏳ Bắt đầu tối ưu hóa tham số...")
    start_time = time.time()
    random_search_rf.fit(X_train_processed, y_train)
    tuning_time = time.time() - start_time
    print("✅ Tối ưu hoàn tất!")

    best_rf_params = random_search_rf.best_params_
    best_rf_cv_score = random_search_rf.best_score_
    rf_tuned_model = random_search_rf.best_estimator_

    y_pred_val = rf_tuned_model.predict(X_val_processed)
    accuracy_val = accuracy_score(y_val, y_pred_val)

    # Ghi log chi tiết
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/tune_random_forest_report.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"📘 BÁO CÁO TỐI ƯU HÓA RANDOM FOREST\n🕒 Thời gian: {timestamp}\n")
        f.write(f"⏱️ Tổng thời gian tuning: {tuning_time:.2f} giây\n\n")

        f.write("📊 KẾT QUẢ CÁC LẦN THỬ:\n")
        results_df = pd.DataFrame(random_search_rf.cv_results_)
        results_df = results_df.sort_values(by="rank_test_score")
        for idx, row in results_df.iterrows():
            f.write(f"- Lần {idx + 1}:\n")
            f.write(f"  📌 Tham số: {row['params']}\n")
            f.write(f"  🎯 Độ chính xác (CV): {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}\n")
            f.write(f"  ↪️ Xếp hạng: {row['rank_test_score']}\n\n")

        f.write("🏆 TỐT NHẤT:\n")
        f.write(f"✔️ Tham số tốt nhất: {best_rf_params}\n")
        f.write(f"🎯 Độ chính xác tốt nhất (CV): {best_rf_cv_score:.4f}\n\n")

        f.write("🧪 ĐÁNH GIÁ TRÊN TẬP VALIDATION (X_val):\n")
        f.write(f"🎯 Accuracy: {accuracy_val:.4f}\n")
        f.write(classification_report(y_val, y_pred_val))
        f.write("\n")

    # Vẽ ma trận nhầm lẫn
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_val, y_pred_val, labels=y.unique())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
    plt.title("Random Forest Confusion Matrix (Optimized)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    safe_timestamp = timestamp.replace(":", "-")
    plt.savefig(f"figures/confusion_matrix_rf_{safe_timestamp}.png", bbox_inches="tight")
    plt.close()

    # Lưu mô hình
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_tuned_model, f"models/random_forest_best.pkl")
    print("✅ Mô hình đã được lưu vào models/random_forest_best.pkl")
