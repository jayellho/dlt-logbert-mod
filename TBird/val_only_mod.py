import sys
sys.path.append("../")

import torch
import random
from bert_pytorch.bert_mod import BERTTrainer, BERTValidator

def bert_val_preprocess(val_data):
    input_data = []
    mask_data = []
    for seq in val_data:
        mask = [0]*512
        seq = [3] + seq
        seq.append(2)
        mask[:len(seq)] = [1] * len(seq)
        seq = seq + [0] * (512 - len(seq))
        input_data.append(seq)
        mask_data.append(mask)
    return torch.tensor(input_data), torch.tensor(mask_data)

def calculate_threshold(anomaly_scores, percentile=0.001):
    sorted_scores = sorted(anomaly_scores)
    index = int((1 - percentile) * len(sorted_scores))
    index = min(max(index, 0), len(sorted_scores) - 1)
    threshold = sorted_scores[index]
    return threshold

if __name__ == "__main__":
    trainer = BERTTrainer()
    model_save_path = './output/bert_trained_model.pth'
    trainer.model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    validator = BERTValidator(model=trainer.model)

    with open('./output/hdfs/train', 'r') as file:
        content = file.read()
    train = [list(map(int, line.split())) for line in content.splitlines()]
    train = [[item + 4 for item in sublist] for sublist in train]
    split_ind = int(0.8*len(train))
    val_data = train[split_ind:]
    val_data, val_mask = bert_val_preprocess(val_data)
    anomaly_scores = validator.validate_batch(val_data, val_mask)   
    print(anomaly_scores)
    print(calculate_threshold(anomaly_scores))