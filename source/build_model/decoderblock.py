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
    
# benmark
def benchmark_decoder_blocks(embed_dim=512, num_heads=8, ffn_hidden_dim=2048, 
                            seq_len=100, batch_size=32, num_iterations=100, device='cuda'):
    """
    Benchmark so sánh Custom DecoderBlock với PyTorch TransformerDecoderLayer
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_hidden_dim: FFN hidden dimension
        seq_len: Sequence length
        batch_size: Batch size
        num_iterations: Number of iterations for benchmarking
        device: 'cuda' hoặc 'cpu'
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK DECODER BLOCKS")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Embed Dim: {embed_dim}")
    print(f"  - Num Heads: {num_heads}")
    print(f"  - FFN Hidden Dim: {ffn_hidden_dim}")
    print(f"  - Seq Len: {seq_len}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Device: {device}")
    print(f"  - Iterations: {num_iterations}")
    print(f"{'='*70}\n")
    
    # Khởi tạo dữ liệu
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    encoder_output = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Custom DecoderBlock
    custom_decoder = DecoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=0.1
    ).to(device)
    
    # PyTorch TransformerDecoderLayer
    pytorch_decoder = nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ffn_hidden_dim,
        dropout=0.1,
        batch_first=True,
        device=device
    )
    
    # Causal mask cho self-attention
    causal_mask = nn.Transformer.generate_square_subsequent_mask(
        seq_len, device=device, dtype=torch.float32
    )
    
    # Benchmark Forward Pass
    print("1. FORWARD PASS BENCHMARK")
    print("-" * 70)
    
    # Custom Decoder Forward
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = custom_decoder(x, encoder_output)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    custom_forward_time = time.time() - start_time
    custom_forward_avg = custom_forward_time / num_iterations
    
    # PyTorch Decoder Forward
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = pytorch_decoder(x, encoder_output, tgt_mask=causal_mask)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    pytorch_forward_time = time.time() - start_time
    pytorch_forward_avg = pytorch_forward_time / num_iterations
    
    print(f"Custom Decoder:")
    print(f"  - Total time: {custom_forward_time:.4f}s")
    print(f"  - Average per iteration: {custom_forward_avg*1000:.4f}ms")
    
    print(f"\nPyTorch Decoder:")
    print(f"  - Total time: {pytorch_forward_time:.4f}s")
    print(f"  - Average per iteration: {pytorch_forward_avg*1000:.4f}ms")
    
    speedup = pytorch_forward_time / custom_forward_time
    print(f"\nSpeedup (Custom vs PyTorch): {speedup:.2f}x")
    
    # Benchmark Memory Usage
    print(f"\n2. MEMORY USAGE BENCHMARK")
    print("-" * 70)
    
    def get_memory_usage(model, x, encoder_output, iterations=10):
        process = psutil.Process(os.getpid())
        
        torch.cuda.empty_cache() if device == 'cuda' else None
        
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
        else:
            start_mem = process.memory_info().rss / 1024 / 1024
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x, encoder_output) if hasattr(model, 'self_mha') else model(x, encoder_output, tgt_mask=causal_mask)
        
        if device == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated()
            memory_used = (peak_mem - start_mem) / 1024 / 1024
        else:
            end_mem = process.memory_info().rss / 1024 / 1024
            memory_used = end_mem - start_mem
        
        return memory_used
    
    custom_mem = get_memory_usage(custom_decoder, x, encoder_output, iterations=10)
    pytorch_mem = get_memory_usage(pytorch_decoder, x, encoder_output, iterations=10)
    
    print(f"Custom Decoder:")
    print(f"  - Peak memory allocated: {custom_mem:.2f}MB")
    
    print(f"\nPyTorch Decoder:")
    print(f"  - Peak memory allocated: {pytorch_mem:.2f}MB")
    
    mem_improvement = ((pytorch_mem - custom_mem) / pytorch_mem * 100) if pytorch_mem > 0 else 0
    print(f"\nMemory efficiency: {mem_improvement:+.2f}%")
    
    # Parameter count
    print(f"\n3. MODEL PARAMETERS")
    print("-" * 70)
    
    custom_params = sum(p.numel() for p in custom_decoder.parameters())
    pytorch_params = sum(p.numel() for p in pytorch_decoder.parameters())
    
    print(f"Custom Decoder: {custom_params:,} parameters")
    print(f"PyTorch Decoder: {pytorch_params:,} parameters")
    
    # Profiling
    print(f"\n4. DETAILED PROFILING (Custom Decoder)")
    print("-" * 70)
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == 'cuda' else [ProfilerActivity.CPU],
                 record_shapes=True) as prof:
        with record_function("Custom Decoder Forward"):
            with torch.no_grad():
                for _ in range(5):
                    _ = custom_decoder(x, encoder_output)
    
    print(prof.key_averages().table(sort_by="cpu_time_total" if device == 'cpu' else "cuda_time_total", row_limit=10))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Forward Pass Speed: Custom is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"✓ Memory Efficiency: Custom is {abs(mem_improvement):.2f}% {'better' if mem_improvement > 0 else 'worse'}")
    print(f"✓ Parameter Count: Custom has {custom_params:,} vs PyTorch {pytorch_params:,}")
    print(f"{'='*70}\n")
    
    return {
        'custom_forward_time': custom_forward_avg,
        'pytorch_forward_time': pytorch_forward_avg,
        'speedup': speedup,
        'custom_memory': custom_mem,
        'pytorch_memory': pytorch_mem,
        'custom_params': custom_params,
        'pytorch_params': pytorch_params
    }

# Chạy benchmark
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = benchmark_decoder_blocks(
        embed_dim=512,
        num_heads=8,
        ffn_hidden_dim=2048,
        seq_len=100,
        batch_size=32,
        num_iterations=100,
        device=device
    )