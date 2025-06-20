import torch
import torch.nn as nn
import json
import numpy as np
import os

MODEL_PATH = "model_service/lstm_model.pt"
NORMALIZATION_PATH = "model_service/normalization.json"
SEQUENCE_LENGTH = 30

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

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None
        self.sequence = []

        self.load_model()

    def load_model(self):
        with open(NORMALIZATION_PATH, "r") as f:
            norm = json.load(f)
            self.mean = np.array(norm["mean"])
            self.std = np.array(norm["std"])

        input_size = len(self.mean)
        self.model = LSTMAutoencoder(input_size=input_size)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

    def predict(self, new_data):
        self.sequence.append(new_data)
        if len(self.sequence) < SEQUENCE_LENGTH:
            return None  # not enough context yet

        self.sequence = self.sequence[-SEQUENCE_LENGTH:]
        window = np.array(self.sequence)
        window = (window - self.mean) / self.std
        window_tensor = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)

        with torch.no_grad():
            output = self.model(window_tensor)
            error = torch.mean((output - window_tensor) ** 2).item()

        return error