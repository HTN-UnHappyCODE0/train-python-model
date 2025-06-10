import logging
import csv
import json
from datetime import datetime
import os

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Cấu hình logging cơ bản
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'train_history.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_to_logfile(message: str):
    logging.info(message)

def log_to_csv(model_name: str, accuracy: float, loss: float, epochs: int):
    csv_path = os.path.join(LOG_DIR, 'train_history.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['timestamp', 'model', 'accuracy', 'loss', 'epochs'])
        writer.writerow([datetime.now(), model_name, accuracy, loss, epochs])

def log_to_json(model_name: str, accuracy: float, loss: float, epochs: int):
    json_path = os.path.join(LOG_DIR, 'train_history.json')
    entry = {
        'timestamp': str(datetime.now()),
        'model': model_name,
        'accuracy': accuracy,
        'loss': loss,
        'epochs': epochs
    }
    with open(json_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
