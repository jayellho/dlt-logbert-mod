import torch
from transformers import BertConfig, BertForMaskedLM, AdamW
from scipy.stats import gmean


class BERTTrainer:
    def __init__(self, vocab_size=100, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12, lr=1e-5):
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

    def validate(self, sequence, mask):
        anomaly_scores = []
        seq_len = sum(mask)
        for i in range(1, seq_len-1):
            original_token_id = sequence[i]
            sequence[i] = 4

            labels = [-100] * 512
            labels[i] = original_token_id

            input_ids = sequence.unsqueeze(0).to(self.device)
            attention_mask = mask.unsqueeze(0).to(self.device)
            labels = torch.tensor([labels], device=self.device)
            # Get the model output
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits  # Shape: [1, sequence_length, vocab_size]

            # Calculate probability of the original token at position i
            token_probs = torch.softmax(logits[0, i], dim=-1)
            token_prob = token_probs[original_token_id].item()

            # Calculate anomaly score (1 - probability)
            anomaly_score = 1 - token_prob
            anomaly_scores.append(anomaly_score)

        # Aggregate anomaly scores using the geometric mean
        sequence_score = gmean(anomaly_scores) if anomaly_scores else 0.0
        return sequence_score

    def validate_batch(self, sequences, masks):
        sequence_scores = []
        for i in range(len(sequences)):
            anomaly_score = self.validate(sequences[i], masks[i])
            sequence_scores.append(anomaly_score)
        return sequence_scores

class BERTDeploy:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def deploy(self, sequence, mask):
        anomaly_scores = []
        seq_len = sum(mask)
        for i in range(1, seq_len-1):
            original_token_id = sequence[i]
            sequence[i] = 4

            labels = [-100] * 512
            labels[i] = original_token_id

            input_ids = sequence.unsqueeze(0).to(self.device)
            attention_mask = mask.unsqueeze(0).to(self.device)
            labels = torch.tensor([labels], device=self.device)

            # Get the model output
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits  # Shape: [1, sequence_length, vocab_size]

            # Calculate probability of the original token at position i
            token_probs = torch.softmax(logits[0, i], dim=-1)
            token_prob = token_probs[original_token_id].item()

            # Calculate anomaly score (1 - probability)
            anomaly_score = 1 - token_prob
            anomaly_scores.append(anomaly_score)

        # Aggregate anomaly scores using the geometric mean
        sequence_score = gmean(anomaly_scores) if anomaly_scores else 0.0
        return sequence_score

    def score_batch(self, sequences, masks):
        sequence_scores = []
        for i in range(len(sequences)):
            anomaly_score = self.deploy(sequences[i], masks[i])
            sequence_scores.append(anomaly_score)
        return sequence_scores