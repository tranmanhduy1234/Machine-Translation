import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedFlashMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, is_causal=False, 
                use_cache=False, k_cache=None, v_cache=None):
        B, T_q, C = query.shape
        T_k = key.size(1)
        
        # === In-projection ===
        if query is key and key is value:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(B, T_q, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
        else:
            w = self.in_proj_weight
            b = self.in_proj_bias
            
            q = F.linear(query, w[:C], b[:C] if b is not None else None)
            q = q.view(B, T_q, self.num_heads, self.head_dim)
            
            if k_cache is None:
                k = F.linear(key, w[C:2*C], b[C:2*C] if b is not None else None)
                k = k.view(B, T_k, self.num_heads, self.head_dim)
            else:
                k_current = F.linear(key, w[C:2*C], b[C:2*C] if b is not None else None)
                k_current = k_current.view(B, key.size(1), self.num_heads, self.head_dim)
                
                k_current = k_current.transpose(1, 2)
                k = torch.cat([k_cache, k_current], dim=-2)
                
            if v_cache is None:
                v = F.linear(value, w[2*C:], b[2*C:] if b is not None else None)
                v = v.view(B, T_k, self.num_heads, self.head_dim)
            else:
                v_current = F.linear(value, w[2*C:], b[2*C:] if b is not None else None)
                v_current = v_current.view(B, value.size(1), self.num_heads, self.head_dim)
                
                v_current = v_current.transpose(1, 2)
                v = torch.cat([v_cache, v_current], dim=-2)

        if k_cache is None and v_cache is None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            q = q.transpose(1, 2)
        
        if key_padding_mask is not None:
            T_mask = key_padding_mask.size(1)
            key_padding_mask = key_padding_mask.view(B, 1, 1, T_mask)
            
        # === FlashAttention kernel ===
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T_q, C)
        attn_output = self.out_proj(attn_output)
        
        new_kv_cache = None
        if use_cache:
            # Đảm bảo k, v đều ở format [B, num_heads, T, head_dim] khi return
            if k.dim() == 4 and k.shape[1] != self.num_heads:
                # [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
                k = k.transpose(1, 2)
            if v.dim() == 4 and v.shape[1] != self.num_heads:
                # [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
                v = v.transpose(1, 2)
            new_kv_cache = (k, v)
                
        return attn_output, new_kv_cache

class FeedForward(nn.Module):
    """MLP / Feed Forward layer"""
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):
    """Một lớp Decoder với Self-Attention, Cross-Attention và FFN"""
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Self-Attention (Causal)
        self.self_attn = OptimizedFlashMHA(embed_dim, num_heads)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-Attention (với encoder output)
        self.cross_attn = OptimizedFlashMHA(embed_dim, num_heads)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed Forward
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x,
        encoder_output=None,
        self_attn_kv_cache=None,
        cross_attn_kv_cache=None,
        use_cache=False,
        key_padding_mask=None
    ):
        """
        Args:
            x: [B, T, C] - decoder input
            encoder_output: [B, T_enc, C] - encoder output (cho cross-attention)
            self_attn_kv_cache: tuple (k, v) hoặc None
            cross_attn_kv_cache: tuple (k, v) hoặc None
            use_cache: bool - có lưu cache hay không
            key_padding_mask: [B, T] - padding mask
            
        Returns:
            output: [B, T, C]
            new_self_attn_cache: tuple hoặc None
            new_cross_attn_cache: tuple hoặc None
        """
        
        # ========== Self-Attention ==========
        x_norm = self.self_attn_norm(x)
        
        if self_attn_kv_cache is not None:
            k_cache, v_cache = self_attn_kv_cache
            self_attn_out, new_self_attn_cache = self.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                is_causal=True,
                use_cache=use_cache,
                k_cache=k_cache,
                v_cache=v_cache
            )
        else:
            self_attn_out, new_self_attn_cache = self.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                is_causal=True,
                use_cache=use_cache
            )
        
        x = x + self.dropout(self_attn_out)
        
        # ========== Cross-Attention (nếu có encoder output) ==========
        if encoder_output is not None:
            x_norm = self.cross_attn_norm(x)
            
            if cross_attn_kv_cache is not None:
                k_cache, v_cache = cross_attn_kv_cache
                cross_attn_out, new_cross_attn_cache = self.cross_attn(
                    x_norm, encoder_output, encoder_output,
                    key_padding_mask=None,
                    is_causal=False,
                    use_cache=use_cache,
                    k_cache=k_cache,
                    v_cache=v_cache
                )
            else:
                cross_attn_out, new_cross_attn_cache = self.cross_attn(
                    x_norm, encoder_output, encoder_output,
                    key_padding_mask=None,
                    is_causal=False,
                    use_cache=use_cache
                )
            
            x = x + self.dropout(cross_attn_out)
        else:
            new_cross_attn_cache = None
        
        # ========== Feed Forward ==========
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x, new_self_attn_cache, new_cross_attn_cache

class Decoder(nn.Module):
    """Decoder stack với nhiều layers"""
    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (vocab_size)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)
        
    def forward(
        self,
        token_ids,
        encoder_output=None,
        kv_caches=None,
        use_cache=False,
        key_padding_mask=None
    ):
        """
        Training mode: token_ids [B, T]
        Inference mode: token_ids [B, 1] với kv_caches
        
        Args:
            token_ids: [B, T]
            encoder_output: [B, T_enc, C]
            kv_caches: list of (self_attn_cache, cross_attn_cache) cho mỗi layer
            use_cache: bool
            key_padding_mask: [B, T]
            
        Returns:
            logits: [B, T, vocab_size]
            new_kv_caches: list of caches
        """
        
        B, T = token_ids.shape
        
        # Embeddings + Positional encoding
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.positional_embedding(positions)
        x = self.dropout(x)
        
        # Initialize caches nếu không có
        if use_cache and kv_caches is None:
            kv_caches = [(None, None) for _ in range(self.num_layers)]
        
        new_kv_caches = []
        
        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            if kv_caches is not None:
                self_attn_cache, cross_attn_cache = kv_caches[i]
            else:
                self_attn_cache, cross_attn_cache = None, None
            
            x, new_self_attn_cache, new_cross_attn_cache = layer(
                x,
                encoder_output=encoder_output,
                self_attn_kv_cache=self_attn_cache,
                cross_attn_kv_cache=cross_attn_cache,
                use_cache=use_cache,
                key_padding_mask=key_padding_mask
            )
            
            if use_cache:
                new_kv_caches.append((new_self_attn_cache, new_cross_attn_cache))
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output logits
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_kv_caches
        else:
            return logits, None


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Config
    vocab_size = 50257
    embed_dim = 512
    num_heads = 8
    num_layers = 4
    ff_dim = 2048
    batch_size = 4
    seq_len = 128
    
    decoder = Decoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim
    ).to(device)
    
    # ===== TRAINING MODE =====
    print("=" * 60)
    print("TRAINING MODE (Full sequence)")
    print("=" * 60)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    encoder_output = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    logits, _ = decoder(token_ids, encoder_output=encoder_output, use_cache=False)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {vocab_size})")
    
    # ===== INFERENCE MODE WITH KV-CACHE =====
    print("\n" + "=" * 60)
    print("INFERENCE MODE (Incremental decoding with KV-Cache)")
    print("=" * 60)
    
    kv_caches = None
    generated_ids = []
    
    # Prompt (initial tokens)
    prompt = torch.randint(0, vocab_size, (1, 10), device=device)
    
    # Process prompt
    _, kv_caches = decoder(prompt, encoder_output=encoder_output[:1], use_cache=True)
    
    # Autoregressive generation
    current_token = torch.tensor([[100]], device=device)  # Start token
    
    for step in range(20):
        logits, kv_caches = decoder(
            current_token,
            encoder_output=encoder_output[:1],
            kv_caches=kv_caches,
            use_cache=True
        )
        
        # Sample next token
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        
        if step % 5 == 0:
            print(f"Step {step}: Generated token {next_token.item()}")
        
        current_token = next_token
    
    print(f"\nGenerated {len(generated_ids)} tokens")
    print(f"Generated IDs: {generated_ids[:10]}...")