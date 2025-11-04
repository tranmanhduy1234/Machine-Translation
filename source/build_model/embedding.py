import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) * self.scale + self.pos_embed(positions)
        return self.dropout(x)