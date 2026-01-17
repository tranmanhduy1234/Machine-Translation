"""
THÀNH PHẦN DECODER LAYER
"""
import torch.nn as nn
from source.build_model.optimizerMultiheadAttention import OptimizedFlashMHA
from  source.build_model.feedForwardNetword import FeedForwardNetwork_standard

# input: [batch_size, seq_len, d_model] -> output: [batch_size, seq_len, d_model]
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout, bias):
        super().__init__()
        self.self_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=bias, dropout_p=dropout)
        self.cross_mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=bias, dropout_p=dropout)
        self.ffn = FeedForwardNetwork_standard(d_model=embed_dim, d_ff=ffn_hidden_dim, activation='gelu', dropout=dropout, bias=bias)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.norm3 = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # kv_cache [batch_size, num_head, seq_len, head_dim]
        self.self_attn_cache = None
        self.cross_attn_cache = None
        
    def reset_cache(self):
        self.self_attn_cache = None
        self.cross_attn_cache = None
        
    def reorder_cache(self, beam_indices):
        if self.self_attn_cache is not None:
            k, v = self.self_attn_cache
            k = k.index_select(0, beam_indices)
            v = v.index_select(0, beam_indices)
            
            self.self_attn_cache = (k, v)
        
        if self.cross_attn_cache is not None:
            k_c, v_c = self.cross_attn_cache
            
            k_c = k_c.index_select(0, beam_indices)
            v_c = v_c.index_select(0, beam_indices)
            
            self.cross_attn_cache = (k_c, v_c)
        
    def forward(self, x, encoder_output, 
                key_padding_mask_tgt, 
                key_padding_mask_src, 
                is_causal_self=True, 
                is_causal_cross=False,
                use_cache=False):

        current_self_attn_cache = self.self_attn_cache if use_cache else None
        current_cross_attn_cache = self.cross_attn_cache if use_cache else None
        
        residual = x
        x = self.norm1(x)
        
        attn_out1, new_self_attn_cache = self.self_mha(x, x, x, 
                                                        key_padding_mask=key_padding_mask_tgt,
                                                        is_causal=is_causal_self,
                                                        use_cache=use_cache,
                                                        kv_cache=current_self_attn_cache)
        x = residual + self.dropout(attn_out1)
        
        residual = x
        x = self.norm2(x)
        attn_out2, new_cross_attn_cache= self.cross_mha(x, encoder_output, encoder_output, 
                                   key_padding_mask=key_padding_mask_src, 
                                   is_causal=is_causal_cross,
                                   use_cache=use_cache,
                                   kv_cache=current_cross_attn_cache)
        
        x = residual + self.dropout(attn_out2)
        
        residual = x
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        if use_cache:
            self.self_attn_cache = new_self_attn_cache
            self.cross_attn_cache = new_cross_attn_cache
            
        return x