import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Thiết lập logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "preprocess.log")

logging.basicConfig(
    filename=log_file,
    encoding='utf-8',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
        logging.info("🚀 === Bắt đầu xử lý dữ liệu ===")

        # Đường dẫn
        file_path = "./data/raw/train.csv"
        test_file_path = "./data/raw/test.csv"
        sample_submission_path = "./data/raw/sample_submission.csv"
        output_path = "./data/processed/train_selected_cleaned.csv"
        heatmap_path = "./results/heatmap_selected_features_1.png"
        matrix_path = "./results/correlation_matrix.png"

        # Tạo thư mục output nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)


        # Đọc dữ liệu
        df = pd.read_csv(file_path)
        df_test = pd.read_csv(test_file_path)
        df_submission = pd.read_csv(sample_submission_path)

        logging.info(f"📄 Đã đọc train.csv: {df.shape[0]} dòng, {df.shape[1]} cột")
        logging.info(f"📄 Đã đọc test.csv: {df_test.shape[0]} dòng, {df_test.shape[1]} cột")
        logging.info(f"📄 Đã đọc sample_submission.csv: {df_submission.shape[0]} dòng, {df_submission.shape[1]} cột")

        # Tạo đặc trưng mới - Age_Group
        age_bins = [0, 12.9, 19.9, 29.9, 44.9, 59.9, np.inf]
        age_labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior']

        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        df_test['Age_Group'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels)

        logging.info("🧠 Đã tạo đặc trưng mới 'Age_Group' dựa trên cột 'Age'")

        # In vài dòng ví dụ từ train: id, Age, Age_Group
        sample_preview = df[['Age', 'Age_Group']].head()
        logging.info("🔍 Một vài dòng mẫu từ train.csv với 'Age' và 'Age_Group':\n" + sample_preview.to_string(index=False))

                # ====== PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) ======
        logging.info("🔍 === Bắt đầu phân tích khám phá dữ liệu (EDA) ===")

        # Kiểm tra cấu trúc
        logging.info(f"📊 Thông tin DataFrame:\n{df.info()}")

        # Kiểm tra giá trị thiếu
        missing_values = df.isnull().sum()
        logging.info(f"❓ Số lượng giá trị thiếu:\n{missing_values[missing_values > 0]}")

        # Kiểm tra phân bố biến mục tiêu (target)
        TARGET_COLUMN = 'NObeyesdad'
        target_counts = df[TARGET_COLUMN].value_counts()
        logging.info(f"🎯 Phân bố biến mục tiêu '{TARGET_COLUMN}':\n{target_counts}")

        # Biểu đồ phân bố lớp target
        plt.figure(figsize=(8, 5))
        sns.countplot(x=TARGET_COLUMN, data=df, order=target_counts.index)
        plt.xticks(rotation=45)
        plt.title("Phân bố các lớp của biến mục tiêu")
        plt.tight_layout()
        plt.savefig("./results/target_distribution.png")
        plt.close()
        logging.info("📈 Đã lưu biểu đồ phân bố biến mục tiêu tại ./results/target_distribution.png")

        # Loại bỏ các cột không cần thiết cho phân tích
        df_eda = df.drop(columns=['id','Age'])  # Nếu có 'id' thì thêm vào đây: ['id', 'Age']
        features = df_eda.drop(columns=[TARGET_COLUMN]).columns.tolist()

        # Phân loại features
        numerical_cols_eda = df_eda[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols_eda = df_eda[features].select_dtypes(include=['object', 'category']).columns.tolist()
        if 'Age_Group' in df_eda.columns and df_eda['Age_Group'].dtype.name == 'category':
            categorical_cols_eda.append('Age_Group')

        logging.info(f"🔢 Cột số (numerical): {numerical_cols_eda}")
        logging.info(f"🔤 Cột phân loại (categorical): {categorical_cols_eda}")

        # Histogram cho từng biến số
        for col in numerical_cols_eda:
            plt.figure()
            sns.histplot(df_eda[col], kde=True, bins=30)
            plt.title(f"Phân phối {col}")
            plt.tight_layout()
            plt.savefig(f"./results/hist_{col}.png")
            plt.close()
            logging.info(f"📊 Đã lưu histogram cho '{col}'")

        # Boxplot so với target
        for col in numerical_cols_eda:
            plt.figure()
            sns.boxplot(x=TARGET_COLUMN, y=col, data=df_eda)
            plt.xticks(rotation=45)
            plt.title(f"{col} vs {TARGET_COLUMN}")
            plt.tight_layout()
            plt.savefig(f"./results/boxplot_{col}.png")
            plt.close()
            logging.info(f"📦 Đã lưu boxplot '{col} vs {TARGET_COLUMN}'")

        # Countplot cho cột phân loại theo target
        for col in categorical_cols_eda:
            plt.figure(figsize=(8, 5))
            order = df_eda[col].cat.categories if df_eda[col].dtype.name == 'category' else df_eda[col].unique()
            sns.countplot(x=col, hue=TARGET_COLUMN, data=df_eda, order=order)
            plt.xticks(rotation=45)
            plt.title(f"{col} theo từng lớp của {TARGET_COLUMN}")
            plt.tight_layout()
            plt.savefig(f"./results/countplot_{col}.png")
            plt.close()
            logging.info(f"📊 Đã lưu countplot cho '{col}' phân theo {TARGET_COLUMN}")

        # Mã hóa TARGET_COLUMN để đưa vào ma trận tương quan
        df_eda_encoded = df_eda.copy()
        if  df_eda_encoded[TARGET_COLUMN].dtype == "object":
            df_eda_encoded[TARGET_COLUMN] = LabelEncoder().fit_transform(df_eda_encoded[TARGET_COLUMN])

        # Tính ma trận tương quan
        correlation_matrix = df_eda_encoded[numerical_cols_eda].corr()

        plt.figure(figsize=(max(14, len(numerical_cols_eda)*0.8), max(10, len(numerical_cols_eda)*0.6)))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, annot_kws={"size": 8})
        plt.title("Ma trận tương quan giữa các biến số", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("./results/correlation_matrix.png")
        plt.close()

        logging.info("🧮 Đã lưu heatmap ma trận tương quan tại ./results/correlation_matrix.png")
 
        logging.info("✅ === Phân tích khám phá dữ liệu hoàn tất ===")


                # ====== CHỌN CỘT CẦN THIẾT ======
        selected_columns = [
            'Gender', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
            'FAF', 'TUE', 'CALC', 'MTRANS', 'Age_Group', 'NObeyesdad'
        ]
        df_selected = df[selected_columns]
        logging.info(f"✅ Đã chọn {len(selected_columns)} cột cần thiết: {selected_columns}")

        # ====== XOÁ DÒNG THIẾU DỮ LIỆU ======
        before_drop = df_selected.shape[0]
        df_cleaned = df_selected.dropna()
        after_drop = df_cleaned.shape[0]
        logging.info(f"🧹 Đã loại bỏ {before_drop - after_drop} dòng thiếu dữ liệu")

        # ====== MÃ HÓA CÁC CỘT PHÂN LOẠI ======
        label_encoder = LabelEncoder()
        categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns

        for col in categorical_columns:
            original_values = list(df_cleaned[col].unique())
            df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
            encoded_values = list(df_cleaned[col].unique())
            logging.info(f"🔄 Cột '{col}': mã hóa từ {original_values} thành {encoded_values}")

        logging.info(f"✅ Mã hóa hoàn tất cho các cột: {list(categorical_columns)}")

        # ====== LƯU DỮ LIỆU ĐÃ XỬ LÝ ======
        df_cleaned.to_csv(output_path, index=False)
        logging.info(f"📁 Đã lưu dữ liệu đã xử lý tại: {output_path}")

        # ====== VẼ HEATMAP TƯƠNG QUAN ======
        plt.figure(figsize=(14, 10))
        sns.heatmap(df_cleaned.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap of Selected Features")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        logging.info(f"🖼️ Đã lưu heatmap tại: {heatmap_path}")

        # ====== TÁCH DỮ LIỆU TRAIN/VAL và LƯU FILE .PKL ======
        from sklearn.model_selection import train_test_split
        import joblib

        X = df_cleaned.drop(columns=["NObeyesdad"])
        y = df_cleaned["NObeyesdad"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        os.makedirs("data", exist_ok=True)
        joblib.dump(X_train, "data/X_train.pkl")
        joblib.dump(X_val, "data/X_val.pkl")
        joblib.dump(y_train, "data/y_train.pkl")
        joblib.dump(y_val, "data/y_val.pkl")
        joblib.dump(X.columns.tolist(), "data/feature_names.pkl")

        logging.info("✅ Đã tách dữ liệu train/val và lưu file: X_train.pkl, X_val.pkl, y_train.pkl, y_val.pkl")

        # ====== XỬ LÝ FILE TEST ======
        logging.info("🚀 === Bắt đầu xử lý dữ liệu test.csv ===")

        test_output_path = "./data/processed/test_selected_cleaned.csv"
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

        df_test_selected = df_test[selected_columns[:-1]]  # Không có 'NObeyesdad' trong test

        before_test_drop = df_test_selected.shape[0]
        df_test_cleaned = df_test_selected.dropna()
        after_test_drop = df_test_cleaned.shape[0]
        logging.info(f"🧹 Đã loại bỏ {before_test_drop - after_test_drop} dòng thiếu dữ liệu trong test.csv")

        for col in categorical_columns:
            if col in df_test_cleaned.columns:
                try:
                    df_test_cleaned[col] = label_encoder.fit(df[col]).transform(df_test_cleaned[col])
                    logging.info(f"🔄 Đã mã hóa cột '{col}' trong test.csv theo giá trị từ train.csv")
                except Exception as e:
                    logging.warning(f"⚠️ Không thể mã hóa cột '{col}' trong test.csv: {e}")

        df_test_cleaned.to_csv(test_output_path, index=False)
        logging.info(f"📁 Đã lưu dữ liệu test đã xử lý tại: {test_output_path}")

        logging.info("✅ === Xử lý dữ liệu test.csv hoàn tất ===")
        logging.info("🎉 === QUY TRÌNH TIỀN XỬ LÝ HOÀN TẤT ===")

    except Exception as e:
        logging.error("❌ Đã xảy ra lỗi trong quá trình xử lý", exc_info=True)
        print("⚠️ Có lỗi xảy ra. Vui lòng kiểm tra file log tại:", log_file)

if __name__ == "__main__":
    main()