import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging
import os

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
        heatmap_path = "./results/heatmap_selected_features.png"

        # Tạo thư mục output nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)

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

        # Chọn cột
        selected_columns = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
            'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
        ]
        df_selected = df[selected_columns]
        logging.info(f"✅ Đã chọn {len(selected_columns)} cột cần thiết")

        # Xoá dòng thiếu dữ liệu
        before_drop = df_selected.shape[0]
        df_cleaned = df_selected.dropna()
        after_drop = df_cleaned.shape[0]
        logging.info(f"🧹 Đã loại bỏ {before_drop - after_drop} dòng thiếu dữ liệu")

        # Mã hóa các cột phân loại
        label_encoder = LabelEncoder()
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            original_values = list(df_cleaned[col].unique())
            df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
            encoded_values = list(df_cleaned[col].unique())
            logging.info(f"🔄 Cột '{col}': mã hóa từ {original_values} thành {encoded_values}")

        logging.info(f"✅ Mã hóa hoàn tất cho các cột: {list(categorical_columns)}")

        # Lưu dữ liệu đã xử lý
        df_cleaned.to_csv(output_path, index=False)
        logging.info(f"📁 Đã lưu dữ liệu đã xử lý tại: {output_path}")

        # Vẽ heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(df_cleaned.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap of Selected Features")

        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        logging.info(f"🖼️ Đã lưu heatmap tại: {heatmap_path}")

        logging.info("✅ === Xử lý dữ liệu hoàn tất ===\n")

    except Exception as e:
        logging.error("❌ Đã xảy ra lỗi trong quá trình xử lý", exc_info=True)
        print("⚠️ Có lỗi xảy ra. Vui lòng kiểm tra file log tại:", log_file)

if __name__ == "__main__":
    main()
