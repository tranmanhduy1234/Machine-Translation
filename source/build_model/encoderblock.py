"""
THÀNH PHẦN ENCODER LAYER
"""
import torch
import torch.nn as nn
from optimizerMultiheadAttention import OptimizedFlashMHA
from feedForwardNetword import FeedForwardNetwork_standard
import time

# input: [batch_size, seq_len, d_model] -> output: [batch_size, seq_len, d_model]
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout):
        super().__init__()
        self.mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=False, dropout=dropout)
        self.ffn = FeedForwardNetwork_standard(d_model=embed_dim, d_ff=ffn_hidden_dim, activation='swish', dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask):
        x = self.norm1(x)
        mha_out = self.mha(x, x, x, key_padding_mask=key_padding_mask, is_causal=False)
        x = x + self.dropout(mha_out)
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return x