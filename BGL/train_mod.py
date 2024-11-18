import sys
sys.path.append("../")

import torch
import random
from torch.utils.data import Dataset, DataLoader
import math
from bert_pytorch.bert_mod import BERTTrainer, BERTValidator

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

def bert_train_preprocess(train_data):
    masked_data = []
    labels = []
    attention_mask = []
    for seq in train_data:
        original_seq = seq[:]
        lbls = [-100]*300
        mask = [0]*300

        num_to_mask = math.ceil(0.15 * len(seq))
        mask_idx = random.sample(range(len(seq)), num_to_mask)
        for i in mask_idx:
            lbls[i] = original_seq[i]
            prob = random.random()
            if prob < 0.8:
                seq[i] = 4
            elif prob < 0.9:
                seq[i] = random.randrange(5, 51)
        seq = [3] + seq
        seq.append(2)

        mask[:len(seq)] = [1] * len(seq)
        seq = seq + [0] * (300 - len(seq))
        masked_data.append(seq)
        labels.append(lbls)
        attention_mask.append(mask)
    masked_data = torch.tensor(masked_data)
    labels = torch.tensor(labels)
    attention_mask = torch.tensor(attention_mask)
    return {'masked_data': masked_data, 'labels_data': labels, 'attention_mask': attention_mask}

def bert_val_preprocess(val_data):
    input_data = []
    mask_data = []
    for seq in val_data:
        mask = [0]*300
        seq = [3] + seq
        seq.append(2)

        mask[:len(seq)] = [1] * len(seq)
        seq = seq + [0] * (300 - len(seq))
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
    with open('./output/hdfs/train', 'r') as file: # CHANGE PATH
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

    model_save_path = './output/bert_trained_model.pth'
    torch.save(trainer.model.state_dict(), model_save_path)
    print(f"model saved to {model_save_path}")

    validator = BERTValidator(model=trainer.model)

    anomaly_scores = validator.validate_batch(val_data, val_mask)   
    print(calculate_threshold(anomaly_scores))




