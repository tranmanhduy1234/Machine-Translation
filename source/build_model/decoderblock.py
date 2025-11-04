import torch
import torch.nn as nn
from optimizerMultiheadAttention import OptimizedFlashMHA
from feedForwardNetword import FeedForwardNetwork_standard
import time
import psutil
import os
from torch.profiler import profile, record_function, ProfilerActivity

# input: [batch_size, seq_len, d_model] -> output: [batch_size, seq_len, d_model]
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout):
        super().__init__()
        self.self_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=False)
        self.cross_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=False)
        self.ffn = FeedForwardNetwork_standard(d_model=embed_dim, d_ff=ffn_hidden_dim, activation='swish')
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, encoder_output, key_padding_mask_tgt, key_padding_mask_src):
        x = self.norm1(x)
        attn_out1 = self.self_mha(x, x, x, key_padding_mask=key_padding_mask_tgt, is_causal=True)
        x = x + self.dropout(attn_out1)
        x = self.norm2(x)
        attn_out2 = self.cross_mha(x, encoder_output, encoder_output, key_padding_mask=key_padding_mask_src, is_causal=False)
        x = x + self.dropout(attn_out2)
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return x