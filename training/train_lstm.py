import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import json

DATA_PATH = "data/train_FD001.txt"
MODEL_PATH = "model_service/lstm_model.pt"
NORMALIZATION_PATH = "model_service/normalization.json"
SEQUENCE_LENGTH = 30
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 64

# Load CMAPSS sensor data
def load_cmaps_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            sensors = [float(val) for val in parts[5:]]
            data.append(sensors)
    return np.array(data)

class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = [data[i:i+seq_len] for i in range(len(data) - seq_len)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(h_repeated)
        return decoded

def train_model():
    raw_data = load_cmaps_data(DATA_PATH)
    mean = raw_data.mean(axis=0)
    std = raw_data.std(axis=0) + 1e-8
    norm_data = (raw_data - mean) / std

    with open(NORMALIZATION_PATH, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    split_idx = int(len(norm_data) * 0.8)
    train_seq = SequenceDataset(norm_data[:split_idx], SEQUENCE_LENGTH)
    val_seq = SequenceDataset(norm_data[split_idx:], SEQUENCE_LENGTH)

    train_loader = DataLoader(train_seq, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAutoencoder(input_size=norm_data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model improved. Saved to {MODEL_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()