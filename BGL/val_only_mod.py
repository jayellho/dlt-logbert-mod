import sys
sys.path.append("../")

import torch
import random
from bert_pytorch.bert_mod import BERTTrainer, BERTValidator

import os
from dotenv import load_dotenv


# load params from .env
load_dotenv()
max_seq_len=os.getenv("MAX_SEQ_LEN")

# load paths from .env
data_dir = os.path.expanduser(os.getenv("DATA_DIR"))
output_dir = os.path.expanduser(os.getenv("OUTPUT_DIR"))
log_file = os.path.expanduser(os.getenv("LOG_FILE"))
model_save_path = os.path.expanduser(os.getenv("MODEL_DIR"))

# construct other paths.
train_file = os.path.join(output_dir, 'train')
test_normal_file = os.path.join(output_dir, 'test_normal')
test_abnormal_file = os.path.join(output_dir, 'test_abnormal')

def bert_val_preprocess(val_data, max_seq_len=int(max_seq_len)+2):
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

def calculate_threshold(anomaly_scores, percentile=0.001):
    sorted_scores = sorted(anomaly_scores)
    index = int((1 - percentile) * len(sorted_scores))
    index = min(max(index, 0), len(sorted_scores) - 1)
    threshold = sorted_scores[index]
    return threshold

if __name__ == "__main__":
    trainer = BERTTrainer()
    # model_save_path = './output/bert_trained_model.pth' # TEMP TO DELETE
    trainer.model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    validator = BERTValidator(model=trainer.model)

    with open(train_file, 'r') as file:
        content = file.read()
    train = [list(map(int, line.split())) for line in content.splitlines()]
    train = [[item + 4 for item in sublist] for sublist in train]
    split_ind = int(0.8*len(train))
    val_data = train[split_ind:]
    val_data, val_mask = bert_val_preprocess(val_data)
    anomaly_scores = validator.validate_batch(val_data, val_mask)   
    print(anomaly_scores)
    print(calculate_threshold(anomaly_scores))