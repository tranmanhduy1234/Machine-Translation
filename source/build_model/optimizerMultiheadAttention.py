import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.attention import SDPBackend

class OptimizedFlashMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, bias=True, dropout_p=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        self.drop_out_p = dropout_p
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, val=0.0)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, val=0.0)
            
    # kv_cache: batch_size, numhead, seq_len, d_model
    def forward(self, query, key, value, key_padding_mask=None, is_causal=False,
                use_cache=False, kv_cache=None):
        B, T, D = query.shape
        src_len = key.size(1)
        
        # ======= Self-Attention =======
        if query is key and key is value:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(B, T, 3, self.num_head, self.head_dim)
            q, k, v = qkv.unbind(dim=2) # batch_size, seqlen, num_head, head_dim
            
            q = q.transpose(1, 2).contiguous() # [batch_size, numheam, seqlen, head_dim]
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            if use_cache:
                if kv_cache is not None:
                    k_past, v_past = kv_cache
                    k = torch.cat((k_past, k), dim=2)
                    v = torch.cat((v_past, v), dim=2)
                kv_cache = (k, v)
        # ======= Cross-Attention =======
        else:
            w = self.in_proj_weight
            b = self.in_proj_bias
            
            q = F.linear(query, w[:D], b[:D] if b is not None else None)
            q = q.view(B, T, self.num_head, self.head_dim)
            q = q.transpose(1, 2).contiguous()
            
            if kv_cache is not None and use_cache:
                k, v = kv_cache
            else:                
                k = F.linear(key, w[D:2*D], b[D:2*D] if b is not None else None)
                v = F.linear(value, w[2*D:], b[2*D:] if b is not None else None)
                
                k = k.view(B, src_len, self.num_head, self.head_dim)
                v = v.view(B, src_len, self.num_head, self.head_dim)
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                if use_cache:
                    kv_cache = (k, v)
                    
        src_len = k.size(2)
        
        # Có 1 cái hay đó là khi inference, đối với các trường hợp có promt, trong lần đầu chạy ta vẫn sẽ phải có causal mask, kể cả có cache hay không
        # Các lần sau đó có cache, thì sẽ ko cần causal, bởi ta chỉ duy trì đúng 1 phần tử query duy nhất.
        attn_mask = self._create_mask(T=T, src_len=src_len, 
                                      is_causal=is_causal, 
                                      key_padding_mask=key_padding_mask, 
                                      device=query.device)
        
        with torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.drop_out_p if self.training else 0.0
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, kv_cache
    
    def _create_mask(self, T, src_len, is_causal, key_padding_mask, device):
        if is_causal and key_padding_mask is None:
            causal_mask = torch.tril(
                torch.ones(T, src_len, dtype=torch.bool, device=device)
            )
            return causal_mask.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, T, src_len)
        elif is_causal and key_padding_mask is not None:
            causal_mask = torch.tril(
                torch.ones(T, src_len, dtype=torch.bool, device=device)
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, src_len)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, src_len)
            combined_mask = causal_mask & padding_mask
            return combined_mask.contiguous()
        elif not is_causal and key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).contiguous()  # (B, 1, 1, src_len)
            return padding_mask
        return None
