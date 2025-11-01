import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardNetwork_standard(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class GLUFFNVariant(nn.Module):
    """
    Gated Linear Unit FFN
    Tốt hơn: Linear -> Activation -> Dropout -> Linear * 2 (gate)
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear_gate = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        FFN with gating mechanism (như GLU)
        Công thức: output = (linear1(x) * activation) ⊙ gate(linear1(x))
        """
        gate = self.linear1(x)
        x = self.activation(gate)
        x = self.dropout(x)
        x = self.linear2(x) * torch.sigmoid(self.linear_gate(gate))
        x = self.dropout(x)
        return x

class GeGLUFFN(nn.Module):
    """
    GeGLU: Gated GLU variant
    Công thức: x * GELU(W*x + b) - tốt hơn GLU
    Được dùng trong T5, PaLM
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x -> [linear1 split thành 2] -> GELU trên nửa 2 -> nhân với nửa 1
        """
        x_proj = self.linear1(x)
        x_proj, gates = x_proj.chunk(2, dim=-1)
        x = x_proj * F.gelu(gates)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN
    Có k experts khác nhau, router quyết định expert nào dùng
    """
    def __init__(self, d_model=512, d_ff=2048, num_experts=4, dropout=0.1, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Tạo k experts FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])
        
        # Router: quyết định expert nào
        self.router = nn.Linear(d_model, num_experts)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Router output: (batch, seq_len, num_experts)
        Chọn top_k experts có score cao nhất
        """
        # Router scores
        router_logits = self.router(x)  # (B, T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Chọn top_k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # normalize
        
        # Expert computation
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (B, T, d_model, num_experts)
        
        # Gather outputs từ selected experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            # Gather: lấy output của expert thứ top_k_indices[:, :, i]
            expert_output = torch.gather(
                expert_outputs, -1,
                top_k_indices[:, :, i:i+1].unsqueeze(-2).expand(-1, -1, x.shape[-1], -1)
            ).squeeze(-1)
            output = output + expert_output * top_k_probs[:, :, i:i+1]
        
        return self.dropout(output)

class OptimizedFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1, activation='gelu', use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU(approximate='tanh')  # Nhanh hơn
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class CompressedFFN(nn.Module):
    """
    Compressed FFN - giảm parameters bằng bottleneck
    d_model -> d_bottleneck -> d_ff -> d_bottleneck -> d_model
    """
    def __init__(self, d_model=512, d_ff=2048, d_bottleneck=256, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_bottleneck)
        self.ffn = nn.Sequential(
            nn.Linear(d_bottleneck, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_bottleneck),
        )
        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.down_proj(x)
        x = self.ffn(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x

# ======== Test & Benchmark ========
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, d_model, d_ff = 16, 512, 512, 2048
    
    x = torch.randn(B, T, d_model, device=device)
    
    models = [
        ("Standard FFN", FeedForwardNetwork(d_model, d_ff)),
        ("GLUFN", GLUFFNVariant(d_model, d_ff)),
        ("GeGLU", GeGLUFFN(d_model, d_ff)),
        ("Optimized FFN", OptimizedFFN(d_model, d_ff)),
        ("Compressed FFN", CompressedFFN(d_model, d_ff, d_bottleneck=256)),
        ("MoE FFN (4 experts)", MoEFFN(d_model, d_ff, num_experts=4)),
    ]
    
    print(f"{'Model':<25} | {'Time (ms)':<12} | {'Memory (MB)':<15} | {'Params':<12}")
    print("-" * 70)
    
    for name, model in models:
        model = model.to(device).eval()
        
        # Count params
        params = sum(p.numel() for p in model.parameters())
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        import time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) / 100 * 1000
        
        if device == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            mem_mb = 0.0
        
        print(f"{name:<25} | {time_ms:>10.2f} | {mem_mb:>13.1f} | {params:>10,}")