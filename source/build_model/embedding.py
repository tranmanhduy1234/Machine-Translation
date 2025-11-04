import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)
        return self.dropout(x)

# ============ DEMO =============
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 32000
    d_model = 512
    max_seq_length = 512
    batch_size = 8
    seq_length = 512
    
    # Tạo embedding layers
    input_embedding = Embedding(vocab_size, d_model, max_seq_length)
    output_embedding = Embedding(vocab_size, d_model, max_seq_length)
    
    # Dummy input (token indices)
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    output_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    input_emb = input_embedding(input_tokens)
    output_emb = output_embedding(output_tokens)
    
    print(f"Input shape: {input_tokens.shape}")
    print(f"Input embedding shape: {input_emb.shape}")
    print(f"Output embedding shape: {output_emb.shape}")
    print(f"\nInput embedding (first 5 values of first token):")
    print(input_emb[0, 0, :5])
    