import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import torch
from bert_pytorch.bert_mod import BERTTrainer, BERTDeploy

def bert_deploy_preprocess(val_data):
    input_data = []
    mask_data = []
    for seq in val_data:
        mask = [0]*512
        seq = [3] + seq
        seq.append(2)
        seq = seq + [0] * (512 - len(seq))
        mask[:len(seq)] = [1] * len(seq)
        input_data.append(seq)
        mask_data.append(mask)
    return torch.tensor(input_data), torch.tensor(mask_data)


if __name__ == "__main__":
    trainer = BERTTrainer()
    model_save_path = './output/bert_trained_model.pth'
    trainer.model.load_state_dict(torch.load(model_save_path))
    deploy = BERTDeploy(model=trainer.model)

    with open('./output/hdfs/test_normal', 'r') as file:
        normal = file.read()
    normal = [list(map(int, line.split())) for line in normal.splitlines()]
    normal = [[item + 4 for item in sublist] for sublist in normal]
    total = len(normal)

    with open('./output/hdfs/test_abnormal', 'r') as file:
        abnormal = file.read()
    abnormal = [list(map(int, line.split())) for line in abnormal.splitlines()]
    abnormal = [[item + 4 for item in sublist] for sublist in abnormal]
    total += len(abnormal)

    normal_data, normal_mask = bert_deploy_preprocess(normal)
    normal_anomaly_scores = deploy.score_batch(normal_data, normal_mask)   

    abnormal_data, abnormal_mask = bert_deploy_preprocess(abnormal)
    abnormal_anomaly_scores = deploy.score_batch(abnormal_data, abnormal_mask)  
    # TODO: save anomaly scores to file to retain for further ablation

    threshold = 0.5 # CHANGE VALUE TO VALUE PRINTED DURING VALIDATION
    normal_positives = sum(1 for value in normal_anomaly_scores if value > threshold)
    abnormal_positives = sum(1 for value in abnormal_anomaly_scores if value > threshold)
    normal_negatives = sum(1 for value in normal_anomaly_scores if value <= threshold)
    abnormal_negatives = sum(1 for value in abnormal_anomaly_scores if value <= threshold)
    precision = abnormal_positives / (abnormal_positives + normal_positives)
    recall = abnormal_positives / (abnormal_positives + abnormal_negatives)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

