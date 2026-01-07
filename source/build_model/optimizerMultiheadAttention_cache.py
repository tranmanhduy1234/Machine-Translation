import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedFlashMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, bias=True, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Gộp 3 projection Q,K,V chung một ma trận để tối ưu cache
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key=None, value=None, key_padding_mask=None, is_causal=False, 
                use_cache=False, kv_cache=None):
        """
        Args:
            query: [B, T_q, C]
            key: [B, T_k, C] hoặc None (self-attention)
            value: [B, T_v, C] hoặc None (self-attention)
            key_padding_mask: [B, T_k]
            is_causal: bool
            use_cache: bool - để lưu KV cache
            kv_cache: tuple của (k_cache, v_cache) hoặc None
        
        Returns:
            attn_output: [B, T_q, C]
            new_kv_cache: tuple hoặc None
        """
        B, T_q, C = query.shape
        
        # Self-attention nếu key và value là None
        if key is None and value is None:
            key = value = query
        
        T_k = key.size(1)
        
        # === In-projection ===
        if key is query and value is query:
            # Self-attention: dùng linear projection chung
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(B, T_q, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            # [B, T, num_heads, head_dim]
        else:
            # Cross-attention
            w = self.in_proj_weight
            b = self.in_proj_bias
            
            q = F.linear(query, w[:C], b[:C] if b is not None else None)
            q = q.view(B, T_q, self.num_heads, self.head_dim)
            
            k = F.linear(key, w[C:2*C], b[C:2*C] if b is not None else None)
            k = k.view(B, T_k, self.num_heads, self.head_dim)
            
            v = F.linear(value, w[2*C:], b[2*C:] if b is not None else None)
            v = v.view(B, T_k, self.num_heads, self.head_dim)
        
        # === KV Cache Processing ===
        if use_cache and kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Cache đã ở format [B, num_heads, past_len, head_dim]
            k = k.transpose(1, 2)  # [B, num_heads, T_k, head_dim]
            v = v.transpose(1, 2)
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
        else:
            # Chuyển về [B, num_heads, T, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        
        # === Xử lý attention mask ===
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k]
            # Chuyển thành [B, 1, 1, T_k] để broadcast
            key_padding_mask = key_padding_mask.view(B, 1, 1, T_k)
        
        # === Scaled Dot-Product Attention (Flash Attention) ===
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # [B, num_heads, T_q, head_dim] -> [B, T_q, num_heads, head_dim] -> [B, T_q, C]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, C)  # Dùng -1 để tự động tính T_q
        attn_output = self.out_proj(attn_output)
        
        # === Return cache nếu cần ===
        new_kv_cache = None
        if use_cache:
            new_kv_cache = (k, v)  # [B, num_heads, seq_len, head_dim]
        
        return attn_output, new_kv_cache


# ============ BENCHMARK ============
if __name__ == "__main__":
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Config
    B, T, C, num_heads = 200, 256, 640, 8
    
    mha = OptimizedFlashMHA(embed_dim=C, num_heads=num_heads).to(device)
    mha.eval()
    
    # ========== Test 1: WITHOUT KV-CACHE ==========
    print("=" * 70)
    print("TEST 1: WITHOUT KV-CACHE (Full sequence recompute)")
    print("=" * 70)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        for step in range(T):
            x = torch.randn(B, step+1, C, device=device)
            attn_out, _ = mha(x, x, x, is_causal=True)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    time_without_cache = time.time() - start
    print(f"Time: {time_without_cache:.4f}s\n")
    
    # ========== Test 2: WITH KV-CACHE ==========
    print("=" * 70)
    print("TEST 2: WITH KV-CACHE (Incremental decoding)")
    print("=" * 70)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    kv_cache = None
    with torch.no_grad():
        for step in range(T):
            x = torch.randn(B, 1, C, device=device)  # Chỉ 1 token mới
            attn_out, kv_cache = mha(x, x, x, use_cache=True, kv_cache=kv_cache)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    time_with_cache = time.time() - start
    print(f"Time: {time_with_cache:.4f}s\n")
    
    # ========== COMPARISON ==========
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Without Cache: {time_without_cache:.4f}s")
    print(f"With Cache:    {time_with_cache:.4f}s")
    print(f"Speedup:       {time_without_cache / time_with_cache:.2f}x")
    print("=" * 70)