"""
LỚP MULTIHEAD ATTENTION FLASH + FUSE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedFlashMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        # gom 3 projection Q,K,V chung một ma trận để tối ưu cache
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, key_padding_mask, is_causal=False):
        B, T, C = query.shape
        src_len = key.size(1)

        # === In-projection ===
        if query is key and key is value:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
        else:
            w = self.in_proj_weight
            b = self.in_proj_bias
            q = F.linear(query, w[:C], b[:C] if b is not None else None)
            k = F.linear(key, w[C:2*C], b[C:2*C] if b is not None else None)
            v = F.linear(value, w[2*C:], b[2*C:] if b is not None else None)
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, src_len, self.num_heads, self.head_dim)
            v = v.view(B, src_len, self.num_heads, self.head_dim)

        # [B, heads, T, head_dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # === Chuẩn hoá mask ===
        # key_padding_mask: (B, src_len) → broadcast đúng chiều [B, 1, 1, src_len]
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B, 1, 1, src_len)
        # === FlashAttention kernel ===
        # Hàm này tự scale QKᵀ / √d, softmax, dropout và nhân V trong GPU kernel
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_padding_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal # mask sẽ được sử dụng sau khi tính score
        )
        # === Output projection ===
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        attn_output = self.out_proj(attn_output)

        return attn_output