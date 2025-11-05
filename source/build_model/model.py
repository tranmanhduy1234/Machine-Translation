import torch
import torch.nn as nn
from decoderblock import DecoderBlock
from embedding import Embedding
from encoderblock import EncoderBlock

class Transformer2025(nn.Module):
    def __init__(self, num_layer_enc = 6, num_layer_dec = 6, d_model = 512, 
                 d_ff = 2048, num_of_heads = 8, dropout_p = 0.1, max_len=512, vocab_size=32000):
        super().__init__()
        # Các hằng số
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        assert self.d_model % self.num_of_heads == 0
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.vocab_size = vocab_size
        self.max_length = max_len
        
        # các thành phần
        # Lớp embedding learnable
        self.embedding = Embedding(self.vocab_size, self.d_model, self.max_length, self.dropout_p)
        # Khối encoder
        self.encoder_component = nn.ModuleList([
            EncoderBlock(embed_dim=self.d_model, num_heads=self.num_of_heads, 
                                               ffn_hidden_dim=self.d_ff, dropout=self.dropout_p)
            for _ in range(num_layer_enc)
        ])
        # Khối decoder
        self.decoder_component = nn.ModuleList([
            DecoderBlock(embed_dim=self.d_model, num_heads=self.num_of_heads, 
                                               ffn_hidden_dim=self.d_ff, dropout=self.dropout_p)
            for _ in range(num_layer_dec)
        ])
        # Lớp Linear cuối cùng 
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.output_projection.weight = self.embedding.token_embed.weight # Buộc trọng số
        # đặt softmax phía ngoài model
    def forward(self, src, tgt, src_kpmask = None, tgt_kpmask=None):
        # src, tgt đều có định dạng [batch_size, seq_len]
        src_embedding = self.embedding(src) # => batch_size, seq_len, d_model
        encoder_output = src_embedding
        # forward encoder
        for encoder_layer in self.encoder_component:
            encoder_output = encoder_layer(encoder_output, key_padding_mask=src_kpmask)
        
        # forward decoder
        tgt_embedding = self.embedding(tgt)
        decoder_output = tgt_embedding
        # forward decoder
        for decoder_layer in self.decoder_component:
            decoder_output = decoder_layer(decoder_output, encoder_output, key_padding_mask_tgt = tgt_kpmask, key_padding_mask_src = src_kpmask)
    
        logits = self.output_projection(decoder_output)
        return logits
    
    def count_parameters(self):
        """Đếm tổng số parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """Lấy device hiện tại của model"""
        return next(self.parameters()).device
    
from torchviz import make_dot

model = Transformer2025().to('cuda')
x = torch.randint(0, 32000, (16, 512), device='cuda')  # ví dụ vocab_size=10000, seq_len=512

Y = model(x, x)
loss = Y.sum()

# visualize before backward
dot = make_dot(loss, params=dict(model.named_parameters()))
dot.render("linear_cuda_graph", format="png")

# compute gradients
loss.backward(retain_graph=True)

print("Grad X:", x.grad)
print("Graph saved as linear_cuda_graph.png")