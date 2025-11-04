import torch
import torch.nn as nn
from optimizerMultiheadAttention import OptimizedFlashMHA
from feedForwardNetword import FeedForwardNetwork_standard
import time

# input: [batch_size, seq_len, d_model] -> output: [batch_size, seq_len, d_model]
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout):
        super().__init__()
        self.mha = OptimizedFlashMHA(embed_dim=embed_dim, num_heads=num_heads, bias=False, dropout=dropout)
        self.ffn = FeedForwardNetwork_standard(d_model=embed_dim, d_ff=ffn_hidden_dim, activation='swish', dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask):
        x = self.norm1(x)
        mha_out = self.mha(x, x, x, key_padding_mask=key_padding_mask, is_causal=False)
        x = x + self.dropout(mha_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x)
        x = x + self.dropout(ffn_out)
        return x
    
if __name__=="__main__":
    batch_size = 16
    seq_len = 1000
    embed_dim = 512
    num_heads = 8
    ffn_hidden_dim = 2048

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Đang sử dụng CPU (Không có GPU)")

    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    block_custom = EncoderBlock(embed_dim, num_heads, ffn_hidden_dim).to(device)
    block_builtin = nn.TransformerEncoderLayer(
        d_model=embed_dim, 
        nhead=num_heads, 
        dim_feedforward=ffn_hidden_dim, 
        batch_first=True
    ).to(device)

    block_custom.eval()
    block_builtin.eval()

    iterations = 1000
    warmup_iterations = 100 # Chạy khởi động để làm nóng GPU

    print(f"Thực hiện {warmup_iterations} lần khởi động và {iterations} lần chạy chính...")

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = block_custom(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start = time.time()
        for _ in range(iterations):
            out_custom = block_custom(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_custom = time.time() - start

        for _ in range(warmup_iterations):
            _ = block_builtin(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None

        start = time.time()
        for _ in range(iterations):
            out_builtin = block_builtin(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_builtin = time.time() - start

    print("-" * 40)
    print(f"Thiết bị: {device}")
    print(f"Kết quả Shape (Custom):   {out_custom.shape}")
    print(f"Kết quả Shape (Built-in): {out_builtin.shape}")
    print("-" * 40)
    print(f"Thời gian Custom:   {t_custom:.6f} giây (cho {iterations} vòng lặp)")
    print(f"Thời gian Built-in: {t_builtin:.6f} giây (cho {iterations} vòng lặp)")
    print("-" * 40)
    if t_builtin > 0:
        print(f"Built-in nhanh hơn: {t_custom / t_builtin:.2f} lần")
    else:
        print("Built-in chạy quá nhanh để đo lường hoặc có lỗi.")