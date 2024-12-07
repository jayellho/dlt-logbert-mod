import torch
from transformers import BertConfig, BertForMaskedLM, AdamW
from scipy.stats import gmean
import os
from dotenv import load_dotenv
import numpy as np

# load params from .env
load_dotenv('../BGL/.env')
max_seq_len=os.getenv("MAX_SEQ_LEN")
max_tokens=os.getenv("MAX_TOKENS")

# helper functions.
def calculate_top_k_percent_gmean(token_probs, k_percent):
    """
    Calculate the geometric mean over the top k percent of token probabilities.

    Args:
        token_probs (list): List of token probabilities for a sequence.
        k_percent (float): Percentage of top probabilities to consider (between 0 and 100).
        
    Returns:
        float: The geometric mean of the top k percent probabilities.
    """
    if not token_probs or k_percent <= 0:
        return 0.0

    # Ensure k_percent is within valid range (0, 100]
    k_percent = min(max(k_percent, 0), 100)

    # Sort probabilities in descending order
    token_probs_sorted = sorted(token_probs, reverse=True)
    
    # Calculate the number of tokens to consider
    k = max(1, int(len(token_probs_sorted) * (k_percent / 100.0)))
    top_k_probs = token_probs_sorted[:k]
    
    # Calculate geometric mean of the top k percent
    return gmean(top_k_probs)

def calculate_harmonic_gmean(token_probs):
    """
    Calculate the harmonic mean of the geometric mean of probabilities and reverse probabilities.
    
    Args:
        token_probs (list): List of token probabilities.
    
    Returns:
        float: Harmonic mean of the geometric means.
    """
    if not token_probs:
        return 0.0

    # Reverse probabilities (1 - p_i for each probability)
    reverse_probs = [1 - p for p in token_probs]

    # Ensure no zero probabilities to avoid math domain errors
    token_probs = np.clip(token_probs, 1e-10, 1.0)
    reverse_probs = np.clip(reverse_probs, 1e-10, 1.0)

    # Calculate geometric means
    gmean_probs = gmean(token_probs)
    gmean_reverse_probs = gmean(reverse_probs)

    # Calculate harmonic mean of geometric means
    harmonic_gmean = (2 * (1-gmean_probs) * gmean_reverse_probs) / ((1-gmean_probs) + gmean_reverse_probs)
    return harmonic_gmean


class BERTTrainer:
    def __init__(self, vocab_size=int(max_tokens)+6, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, max_position_embeddings=int(max_seq_len)+2, type_vocab_size=2, layer_norm_eps=1e-12, lr=1e-4):
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps
        )
        self.model = BertForMaskedLM(self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
    
    def train(self, data_loader, epochs=10):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch in data_loader:
                masked_data = batch['masked_data'].to(self.device)
                labels_data = batch['labels_data'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids=masked_data, attention_mask=attention_mask, labels=labels_data)
                loss = outputs.loss  

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')


class BERTValidator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def validate(self, sequence, mask, experiment_no, k=40): # k pertains to experiment 4.
        anomaly_scores = []

        input_ids = sequence.unsqueeze(0).to(self.device)
        attention_mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        seq_len = sum(mask).item()

        for i in range(1, seq_len - 1):
            original_token_id = sequence[i].item()
            token_probs = torch.softmax(logits[0, i], dim=-1)  # Logits for position i
            token_prob = token_probs[original_token_id].item()  # Probability of original token
            anomaly_scores.append(token_prob)

        
        # experiment 1: sequence score = geometric mean over probability scores. result = 1 - sequence score.
        if experiment_no == 1:
            sequence_score = gmean(anomaly_scores) if anomaly_scores else 0.0
            res = 1 - sequence_score

        # experiment 2: result = geometric mean of (1 - probability) for each token.
        elif experiment_no == 2:
            anomaly_scores = [1-p for p in anomaly_scores]
            sequence_score = gmean(anomaly_scores) if anomaly_scores else 0.0
            res = sequence_score

        # experiment 3: result = harmonic mean of (geometric mean of reverse probabilities + geometric mean of probabilities). formula: harmonic_gmean = (2 * gmean_probs * gmean_reverse_probs) / (gmean_probs + gmean_reverse_probs)
        elif experiment_no == 3:
            sequence_score = calculate_harmonic_gmean(anomaly_scores)
            res = sequence_score

        # experiment 4: experiment 3 but over top k% anomaly scores, where k = 20, 40, 60.
        elif experiment_no == 4:
            k = max(1, len(anomaly_scores) * (k/100))
            top_k_anomaly_scores = sorted(anomaly_scores, reverse=True)[:k]
            sequence_score = calculate_harmonic_gmean(top_k_anomaly_scores)

        # # old experiment 4: sequence score = geometric mean over top k% anomaly scores, where k = 20, 40 and 60. result = 1 - sequence score.
        # sequence_score = calculate_top_k_percent_gmean(anomaly_scores, 20) # CHANGE THE NUMBER HERE.
        # res = 1 - sequence_score

        return res

    def validate_batch(self, sequences, masks):
        sequence_scores = []
        for i in range(len(sequences)):
            anomaly_score = self.validate(sequences[i], masks[i])
            sequence_scores.append(anomaly_score)
            if i % 100 == 0:
                print(f'{i} sequences complete')
        return sequence_scores

class BERTDeploy:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def deploy(self, sequence, mask, experiment_no, k=40): # k only applies for experiment 4 which needs top k% anomaly scores
        anomaly_scores = []

        input_ids = sequence.unsqueeze(0).to(self.device)
        attention_mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        seq_len = sum(mask).item()

        for i in range(1, seq_len - 1):
            original_token_id = sequence[i].item()
            token_probs = torch.softmax(logits[0, i], dim=-1)  # Logits for position i
            token_prob = token_probs[original_token_id].item()  # Probability of original token
            anomaly_scores.append(token_prob)

        # experiment 1: sequence score = geometric mean over probability scores. result = 1 - sequence score.
        if experiment_no == 1:
            sequence_score = gmean(anomaly_scores) if anomaly_scores else 0.0
            res = 1 - sequence_score

        # experiment 2: result = geometric mean of (1 - probability) for each token.
        elif experiment_no == 2:
            anomaly_scores = [1-p for p in anomaly_scores]
            sequence_score = gmean(anomaly_scores) if anomaly_scores else 0.0
            res = sequence_score

        # experiment 3: result = harmonic mean of (geometric mean of reverse probabilities + geometric mean of probabilities). formula: harmonic_gmean = (2 * gmean_probs * gmean_reverse_probs) / (gmean_probs + gmean_reverse_probs)
        elif experiment_no == 3:
            sequence_score = calculate_harmonic_gmean(anomaly_scores)
            res = sequence_score

        # experiment 4: experiment 3 but over top k% anomaly scores, where k = 20, 40, 60.
        elif experiment_no == 4:
            k = max(1, len(anomaly_scores) * (k/100))
            top_k_anomaly_scores = sorted(anomaly_scores, reverse=True)[:k]
            sequence_score = calculate_harmonic_gmean(top_k_anomaly_scores)
        return res

    def score_batch(self, sequences, masks, experiment_no, k=40):
        sequence_scores = []
        for i in range(len(sequences)):
            anomaly_score = self.deploy(sequences[i], masks[i], experiment_no, k)
            sequence_scores.append(anomaly_score)
            if i % 100 == 0:
                print(f'{i} sequences complete')
        return sequence_scores