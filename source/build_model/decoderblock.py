"""
THÀNH PHẦN DECODER LAYER
"""
import torch
import torch.nn as nn
from source.build_model.optimizerMultiheadAttention import OptimizedFlashMHA
from  source.build_model.feedForwardNetword import FeedForwardNetwork_standard

# input: [batch_size, seq_len, d_model] -> output: [batch_size, seq_len, d_model]
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout, bias):
        super().__init__()
        self.self_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=bias)
        self.cross_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForwardNetwork_standard(d_model=embed_dim, d_ff=ffn_hidden_dim, activation='gelu', dropout=dropout, bias=bias)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.norm3 = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, encoder_output, key_padding_mask_tgt, key_padding_mask_src):
        x = self.norm1(x)
        attn_out1= self.self_mha(x, x, x, key_padding_mask=key_padding_mask_tgt, is_causal=True)
        x = x + self.dropout(attn_out1)
        x = self.norm2(x)
        attn_out2 = self.cross_mha(x, encoder_output, encoder_output, key_padding_mask=key_padding_mask_src, is_causal=False)
        x = x + self.dropout(attn_out2)
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return x

class DeccoderBlockCache(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout, bias):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.self_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=bias)
        self.cross_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=bias)
        
        self.ffn = FeedForwardNetwork_standard(d_model=embed_dim, 
                                               d_ff=ffn_hidden_dim, 
                                               activation='gelu', 
                                               dropout=dropout, 
                                               bias=bias)
        
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.norm3 = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # KV-cache layers (tách riêng để compute k, v)
        self.self_attn_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.self_attn_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.self_attn_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.cross_attn_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cross_attn_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cross_attn_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(
        self, x,
        encoder_output,
        key_padding_mask_tgt = None,
        key_padding_mask_src = None,
        self_attn_cache = None,
        cross_attn_cache = None,
        use_cache = False,
        is_incremental = False
    ):
        batch_size = x.shape[0]
        tgt_seq_len = x.shape[1]
        
        # SELF-ATTENTION
        residual = x
        x = self.norm1(x)
        
        if is_incremental and self_attn_cache is not None:
            past_k, past_v = self_attn_cache # [batch, num_heads, past_len, head_dim]
            q = self.self_attn_q_proj(x)
            q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            k = self.self_attn_k_proj(x) # [batch, 1, embed_dim]
            k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            v = self.self_attn_v_proj(x)
            v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            
            new_self_attn_cache = (k, v)
            
            k_full = k.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            v_full = v.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            x_full = x # [batch, 1, embed_dim]
            
            attn_out1 = self._self_attn_with_cache(
                q_input=x_full,
                k_cache=k_full,
                v_cache=v_full,
                key_padding_mask=None,
                is_causal=True
            )
        else:
            attn_out1 = self.self_mha(
                x, x, x,
                key_padding_mask=key_padding_mask_tgt,
                is_causal=True
            )
            
            if use_cache:
                k = self.self_attn_k_proj(x)
                v = self.self_attn_v_proj(x)
                
                k = k.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                new_self_attn_cache = (k, v)
            else:
                new_self_attn_cache = None
                
        x = residual + self.dropout(attn_out1)
        
        # cross attn
        residual = x
        x = self.norm2(x)
        if cross_attn_cache is None and use_cache:
            enc_k = self.cross_attn_k_proj(encoder_output)
            enc_v = self.cross_attn_v_proj(encoder_output)
            
            enc_src_len = encoder_output.shape[1]
            enc_k = enc_k.view(batch_size, enc_src_len, self.num_heads, self.head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, enc_src_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            cross_attn_cache = (enc_k, enc_v)
        
        if cross_attn_cache is not None:
            enc_k, enc_v = cross_attn_cache
            enc_k_full = enc_k.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            enc_v_full = enc_v.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            
            attn_out2 = self._cross_attn_with_cache(
                q_input=x,
                k_cache=enc_k_full,
                v_cache=enc_v_full,
                key_padding_mask=key_padding_mask_src,
                is_causal=False
            )
        else:
            attn_out2 = self.cross_mha(
                x, encoder_output, encoder_output,
                key_padding_mask=key_padding_mask_src,
                is_causal=False
            )
        x = residual + self.dropout(attn_out2)
        
        # FF
        residual = x
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x, new_self_attn_cache, cross_attn_cache
    
    def _self_attn_with_cache(self, q_input, k_cache, v_cache, key_padding_mask, is_causal):
        return self.self_mha(
            q_input,
            k_cache,
            v_cache,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal
        )
    def _cross_attn_with_cache(self, q_input, k_cache, v_cache, key_padding_mask, is_causal):
        return self.cross_mha(
            q_input,
            k_cache,   # Cached encoder keys
            v_cache,   # Cached encoder values
            key_padding_mask=key_padding_mask,
            is_causal=is_causal
        )