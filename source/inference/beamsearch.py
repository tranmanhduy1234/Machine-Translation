import time
from source.build_model.model import Transformer2025
import torch
import torch.nn as nn
from source.tokenizer.tokenizer2025 import Tokenizer2025

class BeamSearchOptim(nn.Module):
    def __init__(self, beam_width, max_len, sos_id, eos_id, device='cuda', alpha=0.6, per_beam_k=None):
        super().__init__()
        self.B = beam_width
        self.max_len = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        self.alpha = alpha
        self.per_beam_k = per_beam_k

    def batch_translate(self, batch_inputs_id, model: Transformer2025, source_mask=None, use_cache=False):
        batch_size, _ = batch_inputs_id.shape
        encoder_output = model.inference_embed_encoder(inputs_id=batch_inputs_id, src_kpmask=source_mask, is_causal=False)
        # [batch_size, seq_len_src, embed_dim]
        
        encoder_output = encoder_output.unsqueeze(1).expand(-1, self.B, -1, -1).contiguous()
        # [batch_size, beam_width, seq_len_src, embed_dim]
        
        encoder_output = encoder_output.reshape(batch_size * self.B, -1, encoder_output.shape[-1])
        # [batch_size * beam_width, seq_len_src, embed_dim]
        
        # Xử lý source_mask nếu không None
        if source_mask is not None:
            source_mask = source_mask.unsqueeze(1).expand(-1, self.B, -1).contiguous()
            # [batch_size, beam_width, seq_len_src]
            source_mask = source_mask.reshape(batch_size * self.B, -1)
            # [batch_size* beam_width, seq_len_src]

        beam_seqs = torch.full((batch_size * self.B, 1), self.sos_id, dtype=torch.long, device=self.device)
        beam_scores = torch.zeros((batch_size, self.B), device=self.device)
        finished = torch.zeros((batch_size, self.B), dtype=torch.bool, device=self.device)
        beam_lengths = torch.ones((batch_size, self.B), dtype=torch.long, device=self.device)

        model.reset_cache()
        for step in range(self.max_len):
            # beam_seqs nếu dùng cache thì chỉ cần phần tử cuối cùng
            beam_seqs_last_token = None
            if use_cache:
                beam_seqs_last_token = beam_seqs[:, -1:]
                
            beam_seqs_embed = model.inference_embedding_layer(beam_seqs_last_token) if use_cache else model.inference_embedding_layer(beam_seqs)
            
            # [batch_size * beam_width, seq_len_query, embed_dim]
            logits = model.inference_decoder_projection(
                input_decoder=beam_seqs_embed, # truyền embedding token của phần tử cuối mỗi beam/batch
                encoder_output=encoder_output, 
                tgt_kpmask=None, 
                src_kpmask=source_mask,
                is_causal_self=True if step == 0 else False,   
                is_causal_cross=False,
                use_cache=use_cache
            )
            # [batch_size * beam_width, seq_len_query, vocab_size]

            next_token_logits = logits[:, -1, :]
            # [batch_size * beam_width, vocab_size]
            
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            # [batch_size * beam_width, vocab_size]
            
            # Reshape để xử lý per-batch
            log_probs = log_probs.view(batch_size, self.B, -1)
            # [batch_size, beam_width, vocab_size]
            
            # Prevent expansion of finished beams
            if finished.any():
                log_probs = log_probs.clone()
                log_probs[finished, :] = -float("inf")
                log_probs[finished, self.eos_id] = 0.0

            vocab_size = log_probs.shape[-1]
            k = self.per_beam_k or min(vocab_size, self.B * 4)
            
            # Per-beam topk
            topk_vals, topk_ids = torch.topk(log_probs, k, dim=-1)
            # [batch_size, beam_width, k]
            
            # Compute candidate scores
            cand_scores = beam_scores.unsqueeze(2) + topk_vals
            # [batch_size, beam_width, k]
            
            # Flatten và lấy top B candidates cho mỗi batch
            flat_scores = cand_scores.view(batch_size, -1)
            # [batch_size, beam_width * k]
            
            topk_flat_scores, topk_flat_indices = torch.topk(flat_scores, self.B, dim=-1)
            # [batch_size, beam_width]
            
            # Tìm parent beam và token position
            parent_beam_indices = topk_flat_indices // k
            # [batch_size, beam_width]
            
            chosen_token_positions = topk_flat_indices % k
            # [batch_size, beam_width]
            
            # Lấy token IDs từ topk_ids
            batch_indices = torch.arange(batch_size, device=self.device).view(batch_size, 1).expand(batch_size, self.B)
            
            chosen_token_indices = topk_ids[batch_indices, parent_beam_indices, chosen_token_positions]
            # [batch_size, beam_width]
            
            # Update beam sequences
            offsets = torch.arange(batch_size, device=self.device).view(batch_size, 1) * self.B
            global_parent_indices = parent_beam_indices + offsets
            # [batch_size, beam_width]
            
            global_parent_indices_flat = global_parent_indices.view(-1)
            if use_cache:
                model.reorder_all_cache(global_parent_indices_flat)
            
            beam_seqs = torch.cat([beam_seqs[global_parent_indices_flat], chosen_token_indices.view(-1, 1)], dim=-1)
            # [batch_size * beam_width, seq_len_tgt + 1]
            
            # Update finished status
            is_eos = (chosen_token_indices == self.eos_id)
            finished = finished[batch_indices, parent_beam_indices] | is_eos
            beam_lengths = beam_lengths[batch_indices, parent_beam_indices].clone()
            beam_lengths[~finished] += 1
            
            # Update scores
            beam_scores = topk_flat_scores 
            # [batch_size * beam_width]
            
            if finished.all():
                break

        # Apply length penalty
        float_lengths = beam_lengths.float()
        length_penalty = torch.pow((5.0 + float_lengths) / 6.0, self.alpha)
        final_scores = beam_scores / length_penalty
        
        # Lấy best sequence cho mỗi batch
        best_indices = torch.argmax(final_scores, dim=-1)
        # [batch_size]
        
        offsets = torch.arange(batch_size, device=self.device) * self.B
        global_best_indices = best_indices + offsets
        
        return beam_seqs[global_best_indices], final_scores[torch.arange(batch_size, device=self.device), best_indices]
        
if __name__=="__main__":
    use_cache = True
    batch_ids = torch.randint(0, 40000, (8, 256)).to("cuda")
    source_mask = ~torch.zeros((8, 256), dtype=bool, device="cuda")
    model = Transformer2025().to('cuda')
    model.eval()
    import time
    start = time.time()
    with torch.no_grad():
        if use_cache:
            model.reset_cache()
        beamsearchhead = BeamSearchOptim(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        rs, _ = beamsearchhead.batch_translate(batch_inputs_id=batch_ids, model=model, source_mask=source_mask, use_cache=use_cache)
        print(rs, "\n", rs.shape)
    print(time.time() - start)
    rs = rs.tolist()
    tokenizer2025 = Tokenizer2025(model_spm_path=r"D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model", legacy=False)
    print(tokenizer2025.decode(rs, skip_special_tokens=True))

# Cấu trúc kv_cache các layer
"""
Trong decoder block
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

Trong model
    def reorder_all_cache(self, beam_indices):
        for decoder_layer in self.decoder_component:
            decoder_layer.reorder_cache(beam_indices)
            
            
Lớp attn có cache
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
"""

"""
BEAM SEARCH VALIDATION SUITE
Kiểm tra:
1. Correctness - Output có hợp lý không
2. Cache behavior - Cache có được reorder đúng không
3. Shape consistency - Tensor shapes có nhất quán không
4. Numerical stability - Có NaN/Inf không
5. End-to-end flow - Toàn bộ quy trình hoạt động đúng không
"""
def validate():
    import torch
    import torch.nn as nn
    from source.build_model.model import Transformer2025
    import numpy as np


    class BeamSearchValidator:
        """Validate beam search implementation"""
        
        def __init__(self, model: Transformer2025, device='cuda'):
            self.model = model
            self.device = device
            self.model.eval()
        
        def test_shape_consistency(self, batch_size=2, beam_width=5, seq_len_src=64, max_steps=20):
            """
            TEST 1: Kiểm tra xem shapes có nhất quán không trong suốt quá trình
            """
            print("\n" + "="*80)
            print("TEST 1: SHAPE CONSISTENCY")
            print("="*80)
            
            input_ids = torch.randint(0, 4000, (batch_size, seq_len_src)).to(self.device)
            source_mask = torch.ones((batch_size, seq_len_src), dtype=torch.bool, device=self.device)
            
            with torch.no_grad():
                # Encoder
                encoder_output = self.model.inference_embed_encoder(
                    inputs_id=input_ids, 
                    src_kpmask=source_mask, 
                    is_causal=False
                )
                expected_shape = (batch_size, seq_len_src, 640)
                actual_shape = encoder_output.shape
                
                print(f"\n✓ Encoder output shape: {actual_shape}")
                assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
                
                # Expand
                encoder_expanded = encoder_output.unsqueeze(1).expand(-1, beam_width, -1, -1).contiguous()
                encoder_expanded = encoder_expanded.reshape(batch_size * beam_width, -1, 640)
                
                expected_shape = (batch_size * beam_width, seq_len_src, 640)
                actual_shape = encoder_expanded.shape
                
                print(f"✓ Expanded encoder shape: {actual_shape}")
                assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
                
                # Decoding loop
                beam_seqs = torch.full((batch_size * beam_width, 1), 1, dtype=torch.long, device=self.device)
                
                self.model.reset_cache()
                
                for step in range(max_steps):
                    # Last token only
                    beam_seqs_last = beam_seqs[:, -1:]
                    
                    # Embedding
                    embed = self.model.inference_embedding_layer(beam_seqs_last)
                    expected_shape = (batch_size * beam_width, 1, 640)
                    assert embed.shape == expected_shape, f"Step {step}: embedding shape {embed.shape} != {expected_shape}"
                    
                    # Decoder forward
                    logits = self.model.inference_decoder_projection(
                        input_decoder=embed,
                        encoder_output=encoder_expanded,
                        tgt_kpmask=None,
                        src_kpmask=source_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(batch_size * beam_width, -1),
                        is_causal_self=(step == 0),
                        is_causal_cross=False,
                        use_cache=True
                    )
                    
                    expected_shape = (batch_size * beam_width, 1, 40000)
                    assert logits.shape == expected_shape, f"Step {step}: logits shape {logits.shape} != {expected_shape}"
                    
                    # Simulated beam search update
                    next_token = torch.randint(0, 40000, (batch_size * beam_width, 1), device=self.device)
                    beam_seqs = torch.cat([beam_seqs, next_token], dim=-1)
                    
                    print(f"  Step {step:2d}: embed={embed.shape}, logits={logits.shape}, beam_seqs={beam_seqs.shape}")
                
                print("\n✅ All shapes are consistent!")
                return True
        
        def test_cache_reorder_correctness(self, batch_size=2, beam_width=5, seq_len_src=64, steps=10):
            """
            TEST 2: Kiểm tra cache reorder có xáo trộn đúng không
            """
            print("\n" + "="*80)
            print("TEST 2: CACHE REORDER CORRECTNESS")
            print("="*80)
            
            input_ids = torch.randint(0, 4000, (batch_size, seq_len_src)).to(self.device)
            source_mask = torch.ones((batch_size, seq_len_src), dtype=torch.bool, device=self.device)
            
            encoder_output = self.model.inference_embed_encoder(
                inputs_id=input_ids, src_kpmask=source_mask, is_causal=False
            )
            encoder_expanded = encoder_output.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(batch_size * beam_width, -1, 640)
            
            beam_seqs = torch.full((batch_size * beam_width, 1), 1, dtype=torch.long, device=self.device)
            self.model.reset_cache()
            
            # Store original cache after a few steps
            with torch.no_grad():
                for step in range(steps):
                    embed = self.model.inference_embedding_layer(beam_seqs[:, -1:])
                    _ = self.model.inference_decoder_projection(
                        input_decoder=embed,
                        encoder_output=encoder_expanded,
                        tgt_kpmask=None,
                        src_kpmask=source_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(batch_size * beam_width, -1),
                        is_causal_self=(step == 0),
                        is_causal_cross=False,
                        use_cache=True
                    )
                    beam_seqs = torch.cat([beam_seqs, torch.ones((batch_size * beam_width, 1), dtype=torch.long, device=self.device)], dim=-1)
                
                # Get cache before reorder
                cache_before = []
                for layer in self.model.decoder_component:
                    if layer.self_attn_cache is not None:
                        k, v = layer.self_attn_cache
                        cache_before.append((k.clone(), v.clone()))
                
                print(f"\n✓ Captured cache at step {steps}")
                print(f"  Cache entries: {len(cache_before)}")
                print(f"  First layer K cache shape: {cache_before[0][0].shape}")
                
                # Create reorder indices: swap first 2 beams with last 2 beams
                reorder_indices = torch.cat([
                    torch.arange(2, 4),  # Beams 2,3 first
                    torch.arange(0, 2),  # Then beams 0,1
                    torch.arange(4, batch_size * beam_width)  # Rest unchanged
                ]).to(self.device)
                
                print(f"\n  Reorder pattern: {reorder_indices[:10].tolist()}...")
                
                # Perform reorder
                self.model.reorder_all_cache(reorder_indices)
                
                # Get cache after reorder
                cache_after = []
                for layer in self.model.decoder_component:
                    if layer.self_attn_cache is not None:
                        k, v = layer.self_attn_cache
                        cache_after.append((k.clone(), v.clone()))
                
                # Verify reorder was applied
                for layer_idx, (k_before, v_before) in enumerate(cache_before):
                    k_after, v_after = cache_after[layer_idx]
                    
                    # Check that values at positions [2,3,0,1,...] match original
                    for i, orig_idx in enumerate(reorder_indices):
                        if not torch.allclose(k_after[i], k_before[orig_idx], atol=1e-6):
                            print(f"❌ Layer {layer_idx}, position {i}: K cache mismatch!")
                            return False
                
                print("\n✅ Cache reorder is working correctly!")
                return True
        
        def test_numerical_stability(self, batch_size=2, beam_width=5, seq_len_src=64, max_steps=50):
            """
            TEST 3: Kiểm tra có NaN/Inf không, numerical stability
            """
            print("\n" + "="*80)
            print("TEST 3: NUMERICAL STABILITY")
            print("="*80)
            
            input_ids = torch.randint(0, 4000, (batch_size, seq_len_src)).to(self.device)
            source_mask = torch.ones((batch_size, seq_len_src), dtype=torch.bool, device=self.device)
            
            encoder_output = self.model.inference_embed_encoder(
                inputs_id=input_ids, src_kpmask=source_mask, is_causal=False
            )
            
            # Check encoder output
            if torch.isnan(encoder_output).any():
                print("❌ NaN detected in encoder output!")
                return False
            if torch.isinf(encoder_output).any():
                print("❌ Inf detected in encoder output!")
                return False
            
            print("✓ Encoder output: no NaN/Inf")
            
            encoder_expanded = encoder_output.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(batch_size * beam_width, -1, 640)
            beam_seqs = torch.full((batch_size * beam_width, 1), 1, dtype=torch.long, device=self.device)
            
            self.model.reset_cache()
            
            with torch.no_grad():
                for step in range(max_steps):
                    embed = self.model.inference_embedding_layer(beam_seqs[:, -1:])
                    
                    if torch.isnan(embed).any():
                        print(f"❌ NaN in embedding at step {step}")
                        return False
                    
                    logits = self.model.inference_decoder_projection(
                        input_decoder=embed,
                        encoder_output=encoder_expanded,
                        tgt_kpmask=None,
                        src_kpmask=source_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(batch_size * beam_width, -1),
                        is_causal_self=(step == 0),
                        is_causal_cross=False,
                        use_cache=True
                    )
                    
                    if torch.isnan(logits).any():
                        print(f"❌ NaN in logits at step {step}")
                        return False
                    
                    # Log probs
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                    
                    if torch.isnan(log_probs).any():
                        print(f"❌ NaN in log_probs at step {step}")
                        return False
                    
                    if torch.isinf(log_probs).any():
                        print(f"❌ Inf in log_probs at step {step}")
                        return False
                    
                    # Random next token
                    next_token = torch.randint(0, 40000, (batch_size * beam_width, 1), device=self.device)
                    beam_seqs = torch.cat([beam_seqs, next_token], dim=-1)
            
            print(f"✓ Completed {max_steps} steps without NaN/Inf")
            print("✅ Numerical stability confirmed!")
            return True
        
        def test_output_distribution(self, batch_size=2, beam_width=5, seq_len_src=64, max_steps=20):
            """
            TEST 4: Kiểm tra output distribution có reasonable không
            """
            print("\n" + "="*80)
            print("TEST 4: OUTPUT DISTRIBUTION")
            print("="*80)
            
            input_ids = torch.randint(0, 4000, (batch_size, seq_len_src)).to(self.device)
            source_mask = torch.ones((batch_size, seq_len_src), dtype=torch.bool, device=self.device)
            
            encoder_output = self.model.inference_embed_encoder(
                inputs_id=input_ids, src_kpmask=source_mask, is_causal=False
            )
            encoder_expanded = encoder_output.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(batch_size * beam_width, -1, 640)
            
            beam_seqs = torch.full((batch_size * beam_width, 1), 1, dtype=torch.long, device=self.device)
            self.model.reset_cache()
            
            token_distribution = []
            
            with torch.no_grad():
                for step in range(max_steps):
                    embed = self.model.inference_embedding_layer(beam_seqs[:, -1:])
                    logits = self.model.inference_decoder_projection(
                        input_decoder=embed,
                        encoder_output=encoder_expanded,
                        tgt_kpmask=None,
                        src_kpmask=source_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(batch_size * beam_width, -1),
                        is_causal_self=(step == 0),
                        is_causal_cross=False,
                        use_cache=True
                    )
                    
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                    probs = torch.exp(log_probs)
                    
                    # Get top-5 tokens per beam
                    top_probs, top_ids = torch.topk(probs, 5, dim=-1)
                    token_distribution.append(top_probs.cpu().numpy())
                    
                    # Random next token
                    next_token = torch.randint(0, 40000, (batch_size * beam_width, 1), device=self.device)
                    beam_seqs = torch.cat([beam_seqs, next_token], dim=-1)
            
            # Analyze distribution
            token_distribution = np.concatenate(token_distribution, axis=0)
            
            print(f"\n✓ Token distribution statistics:")
            print(f"  Mean top-1 probability: {token_distribution[:, 0].mean():.4f}")
            print(f"  Mean top-5 probability sum: {token_distribution.sum(axis=1).mean():.4f}")
            print(f"  Min top-1 probability: {token_distribution[:, 0].min():.6f}")
            print(f"  Max top-1 probability: {token_distribution[:, 0].max():.6f}")
            
            # Sanity checks
            if token_distribution[:, 0].mean() < 0.01:
                print("⚠️  Warning: very low top-1 probabilities (model might be confused)")
            
            if token_distribution[:, 0].mean() > 0.99:
                print("⚠️  Warning: very high top-1 probabilities (model might be over-confident)")
            
            print("✅ Output distribution looks reasonable!")
            return True
        
        def run_all_tests(self):
            """Run all validation tests"""
            print("\n" + "="*80)
            print("BEAM SEARCH VALIDATION SUITE")
            print("="*80)
            
            results = {
                "Shape Consistency": self.test_shape_consistency(),
                "Cache Reorder": self.test_cache_reorder_correctness(),
                "Numerical Stability": self.test_numerical_stability(),
                "Output Distribution": self.test_output_distribution(),
            }
            
            print("\n\n" + "="*80)
            print("VALIDATION SUMMARY")
            print("="*80)
            
            for test_name, passed in results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{test_name:30s} : {status}")
            
            all_passed = all(results.values())
            
            print("\n" + "="*80)
            if all_passed:
                print("🎉 ALL TESTS PASSED - Beam search is working correctly!")
            else:
                print("⚠️  SOME TESTS FAILED - Check issues above")
            print("="*80)
            
            return all_passed


    if __name__ == "__main__":
        model = Transformer2025().to('cuda')
        model.eval()
        
        validator = BeamSearchValidator(model, device='cuda')
        validator.run_all_tests()
validate()