import time
import random
from utils.logger import log_to_logfile, log_to_csv, log_to_json

def train_model():
    # Giả lập thời gian training
    print("Training started...")
    time.sleep(2)

    # Sinh ngẫu nhiên thông số
    accuracy = round(random.uniform(0.8, 0.95), 4)
    loss = round(random.uniform(0.1, 0.4), 4)
    epochs = 10
    model_name = "model_v1"

    print(f"Training completed! Accuracy: {accuracy}, Loss: {loss}")

    # Ghi log
    log_to_logfile(f"Training successful - Accuracy: {accuracy}, Loss: {loss}, Epochs: {epochs}")
    log_to_csv(model_name, accuracy, loss, epochs)
    log_to_json(model_name, accuracy, loss, epochs)

if __name__ == "__main__":
    train_model()
