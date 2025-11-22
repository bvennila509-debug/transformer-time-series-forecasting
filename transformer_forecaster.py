"""transformer_forecaster.py
A lightweight PyTorch Transformer model for time-series forecasting.
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, pred_len=30):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, pred_len)
        )
    def forward(self, src):
        # src: (batch, seq_len, input_dim) -> transform to (seq_len, batch, d_model)
        x = self.input_proj(src)  # (batch, seq_len, d_model)
        x = x.permute(1,0,2)  # (seq_len, batch, d_model)
        x = self.pos_enc(x)
        enc = self.encoder(x)  # (seq_len, batch, d_model)
        # use final time-step encoding for prediction
        final = enc[-1,:,:]  # (batch, d_model)
        out = self.decoder(final)  # (batch, pred_len)
        return out
