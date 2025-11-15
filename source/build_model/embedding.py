"""
THÀNH PHẦN EMBEDDING BAO GỒM TOKEN EMBED + POSITION EMBED
"""
import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)
 
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) * self.scale + self.pos_embed(positions)
        return self.dropout(x)
    
class TUPE(nn.Module):
    def __init__(self, d_model: int = 28, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.absolute_pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.relative_pos_embedding = nn.Embedding(2 * max_seq_len + 1, d_model)
        
        # Parameters cho learnable scaling
        self.rel_weight = nn.Parameter(torch.ones(1))
        self.abs_weight = nn.Parameter(torch.ones(1))
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.absolute_pos_embedding.weight.device
        
        # Absolute positions
        positions = torch.arange(seq_len, device=device)
        abs_embed = self.absolute_pos_embedding(positions)  # (seq_len, d_model)
        
        # Relative positions (tính pairwise)
        pos_indices = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        pos_indices = torch.clamp(pos_indices, -self.max_seq_len, self.max_seq_len) + self.max_seq_len
        rel_embed = self.relative_pos_embedding(pos_indices)  # (seq_len, seq_len, d_model)
        combined = self.abs_weight * abs_embed.unsqueeze(1) + self.rel_weight * rel_embed
        pos_encoding = self.layer_norm(combined.mean(dim=0))  # (seq_len, d_model)
        
        return pos_encoding

class ConvSPE(nn.Module):
    def __init__(self, vocab_size: int = 32000, d_model: int = 512, max_seq_len: int = 512, kernel_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size

        # Token embedding thực tế
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Convolution để học local position patterns
        padding = kernel_size // 2
        self.conv1d = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
        )

        # Relative shift embeddings
        self.shift_embedding = nn.Embedding(2 * kernel_size + 1, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, device=None):
        """
        x: [batch, seq_len] chứa token ids
        """
        if device is None:
            device = x.device

        batch_size, seq_len = x.shape

        # 1. Token embeddings
        token_embed = self.token_embedding(x)  # [batch, seq_len, d_model]

        # 2. Conv1d trên trục seq
        token_embed_t = token_embed.transpose(1, 2)  # [batch, d_model, seq_len]
        conv_output = self.conv1d(token_embed_t)  # [batch, d_model, seq_len]
        conv_embed = conv_output.transpose(1, 2)  # [batch, seq_len, d_model]

        # 3. Shift embeddings (relative)
        shift_indices = torch.arange(-self.kernel_size, self.kernel_size + 1, device=device)
        shift_offset = self.kernel_size
        shift_embed = self.shift_embedding(shift_indices + shift_offset)  # [2*kernel+1, d_model]
        avg_shift = shift_embed.mean(dim=0)  # [d_model]

        # 4. Combine và chuẩn hóa
        pos_encoding = self.layer_norm(conv_embed + avg_shift.unsqueeze(0).unsqueeze(0))
        return pos_encoding  # [batch, seq_len, d_model]

if __name__=="__main__":
    embed_layer = ConvSPE().cuda()
    x = torch.randint(0, 32000, (16, 512), device='cuda')
    out = embed_layer(x)
    print(out.shape)  # (16, 512, 512)