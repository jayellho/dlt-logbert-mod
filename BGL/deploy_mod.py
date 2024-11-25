import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import torch
import random
import csv
from bert_pytorch.bert_mod import BERTTrainer, BERTDeploy
import os
from dotenv import load_dotenv


# load params from .env
load_dotenv()
max_seq_len = os.getenv("MAX_SEQ_LEN")
threshold = float(os.getenv("THRESHOLD"))

# load paths from .env
data_dir = os.path.expanduser(os.getenv("DATA_DIR"))
output_dir = os.path.expanduser(os.getenv("OUTPUT_DIR"))
log_file = os.path.expanduser(os.getenv("LOG_FILE"))
model_save_path = os.path.expanduser(os.getenv("MODEL_DIR"))

# construct other paths.
train_file = os.path.join(output_dir, 'train')
test_normal_file = os.path.join(output_dir, 'test_normal')
test_abnormal_file = os.path.join(output_dir, 'test_abnormal')
normal_anomalies_file = os.path.join(output_dir, 'normal_anomalies.csv')
abnormal_anomalies_file = os.path.join(output_dir, 'abnormal_anomalies.csv')

def bert_deploy_preprocess(val_data, max_seq_len=int(max_seq_len)+2):
    input_data = []
    mask_data = []
    for seq in val_data:
        mask = [0]*max_seq_len
        seq = [3] + seq
        seq.append(2)
        mask[:len(seq)] = [1] * len(seq)
        seq = seq + [0] * (max_seq_len - len(seq))
        input_data.append(seq)
        mask_data.append(mask)
    return torch.tensor(input_data), torch.tensor(mask_data)


if __name__ == "__main__":
    trainer = BERTTrainer()
    # model_save_path = './output/bert_trained_model.pth'
    trainer.model.load_state_dict(torch.load(model_save_path))
    deploy = BERTDeploy(model=trainer.model)

    with open(test_normal_file, 'r') as file:
        normal = file.read()
    normal = [list(map(int, line.split())) for line in normal.splitlines()]
    normal = [[item + 4 for item in sublist] for sublist in normal]
    random.shuffle(normal)
    normal = normal[:4000]
    total = len(normal)

    with open(test_abnormal_file, 'r') as file:
        abnormal = file.read()
    abnormal = [list(map(int, line.split())) for line in abnormal.splitlines()]
    abnormal = [[item + 4 for item in sublist] for sublist in abnormal]
    random.shuffle(abnormal)
    abnormal = abnormal[:4000]
    total += len(abnormal)

    normal_data, normal_mask = bert_deploy_preprocess(normal)
    normal_anomaly_scores = deploy.score_batch(normal_data, normal_mask)   

    abnormal_data, abnormal_mask = bert_deploy_preprocess(abnormal)
    abnormal_anomaly_scores = deploy.score_batch(abnormal_data, abnormal_mask)  
    # TODO: save anomaly scores to file to retain for further ablation

    with open(normal_anomalies_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(normal_anomaly_scores) 
    print("Saved to output.csv")

    with open(abnormal_anomalies_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(abnormal_anomaly_scores) 
    print("Saved to output.csv")

    normal_positives = sum(1 for value in normal_anomaly_scores if value > threshold)
    abnormal_positives = sum(1 for value in abnormal_anomaly_scores if value > threshold)
    normal_negatives = sum(1 for value in normal_anomaly_scores if value <= threshold)
    abnormal_negatives = sum(1 for value in abnormal_anomaly_scores if value <= threshold)
    precision = abnormal_positives / (abnormal_positives + normal_positives)
    recall = abnormal_positives / (abnormal_positives + abnormal_negatives)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

