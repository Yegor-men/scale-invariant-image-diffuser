import torch
from torch import nn


class DummyTextCond(nn.Module):
    def __init__(self, d_channels):
        super().__init__()
        self.text_token_length = 1
        self.text_proj = nn.Linear(in_features=10, out_features=d_channels)
        self.token_norm = nn.LayerNorm(d_channels)

    def forward(self, label_vector):
        tokens = self.text_proj(label_vector).unsqueeze(1)
        tokens = self.token_norm(tokens)
        return tokens
