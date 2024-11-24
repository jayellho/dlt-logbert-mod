import sys
sys.path.append("../")

import torch
import random
from torch.utils.data import Dataset, DataLoader
import math
from bert_pytorch.bert_mod import BERTTrainer, BERTValidator
import os
from dotenv import load_dotenv

# load params from .env
load_dotenv()
max_seq_len=os.getenv("MAX_SEQ_LEN")
max_tokens=os.getenv("MAX_TOKENS")

# load paths from .env
data_dir = os.path.expanduser(os.getenv("DATA_DIR"))
output_dir = os.path.expanduser(os.getenv("OUTPUT_DIR"))
log_file = os.path.expanduser(os.getenv("LOG_FILE"))
model_save_path = os.path.expanduser(os.getenv("MODEL_DIR"))

# construct other paths.
train_file = os.path.join(output_dir, 'train')
test_normal_file = os.path.join(output_dir, 'test_normal')
test_abnormal_file = os.path.join(output_dir, 'test_abnormal')


class BERTDataset(Dataset):
    def __init__(self, data):
        self.masked_data = data['masked_data']
        self.labels_data = data['labels_data']
        self.attention_mask = data['attention_mask']

    def __len__(self):
        return len(self.masked_data)

    def __getitem__(self, idx):
        return {
            'masked_data': self.masked_data[idx],
            'labels_data': self.labels_data[idx],
            'attention_mask': self.attention_mask[idx]
        }

def bert_train_preprocess(train_data, max_seq_len=int(max_seq_len)+2, max_tokens=max_tokens):
    masked_data = []
    labels = []
    attention_mask = []
    for seq in train_data:
        original_seq = seq[:]
        lbls = [-100]*max_seq_len
        mask = [0]*max_seq_len

        num_to_mask = math.ceil(0.15 * len(seq))
        mask_idx = random.sample(range(len(seq)), num_to_mask)
        for i in mask_idx:
            lbls[i] = original_seq[i]
            prob = random.random()
            if prob < 0.8:
                seq[i] = 4
            elif prob < 0.9:
                seq[i] = random.randrange(5, int(max_tokens) + 5)
        seq = [3] + seq
        seq.append(2)

        mask[:len(seq)] = [1] * len(seq)
        seq = seq + [0] * (max_seq_len - len(seq))
        masked_data.append(seq)
        labels.append(lbls)
        attention_mask.append(mask)
    masked_data = torch.tensor(masked_data)
    labels = torch.tensor(labels)
    attention_mask = torch.tensor(attention_mask)
    return {'masked_data': masked_data, 'labels_data': labels, 'attention_mask': attention_mask}

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
    with open(train_file, 'r') as file: # CHANGE PATH
        content = file.read()

    train = [list(map(int, line.split())) for line in content.splitlines()]
    train = [[item + 4 for item in sublist] for sublist in train]
    split_ind = int(0.8*len(train))
    train_data = train[:split_ind]
    val_data = train[split_ind:]

    train_data = bert_train_preprocess(train_data)
    train_dataset = BERTDataset(train_data)
    val_data, val_mask = bert_val_preprocess(val_data)
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print('data prepped')
    trainer = BERTTrainer()
    trainer.train(data_loader, epochs=10) 

    # model_save_path = '/home/jl/.dataset/output/bert_trained_model_BGL.pth' # TO DELETE
    torch.save(trainer.model.state_dict(), model_save_path)
    print(f"model saved to {model_save_path}")

    validator = BERTValidator(model=trainer.model)

    anomaly_scores = validator.validate_batch(val_data, val_mask)   
    print(calculate_threshold(anomaly_scores))




