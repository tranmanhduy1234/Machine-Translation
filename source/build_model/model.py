"""
XÂY DỰNG KIẾN TRÚC
"""
import torch
import torch.nn as nn
from decoderblock import DecoderBlock
from embedding import Embedding
from encoderblock import EncoderBlock

class Transformer2025(nn.Module):
    def __init__(self, num_layer_enc = 6, num_layer_dec = 6, d_model = 512, 
                 d_ff = 2048, num_of_heads = 8, dropout_p = 0.3, max_len=512, vocab_size=32000):
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
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=True)
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
    
    # Nhận đầu vào là batch phần tử đã được tokenizer dạng [batch_size, seq_len]
    def inference_embedding_layer(self, input_embeding):
        return self.embedding(input_embeding) # => trả ra [batch_size, seq_len, embed_dim]
    
    # input_shape: [batch_size, seq_len, embed_dim] -> output: [batch_size, seq_len, embed_dim]
    def inference_encoder_layer(self, input_encoder, src_kpmask):
        encoder_output = input_encoder
        for encoder_layer in self.encoder_component:
            encoder_output = encoder_layer(encoder_output, key_padding_mask=src_kpmask)
        return encoder_output

    # input_shape: [batch_size, seq_len, embed_dim] -> output: [batch_size, seq_len, embed_dim]
    def inference_decoder_layer(self, input_decoder, encoder_output, tgt_kpmask, src_kpmask):
        decoder_output = input_decoder
        for decoder_layer in self.decoder_component:
            decoder_output = decoder_layer(decoder_output, encoder_output, key_padding_mask_tgt = tgt_kpmask, key_padding_mask_src = src_kpmask)
        return decoder_output
    
    # input_shape: [batch_size, seq_len, embed_dim] -> output: [batch_size, seq_len, vocab_size]
    def inference_output_projection(self, output_decoder):
        return self.output_projection(output_decoder) # return logits
    
    # input_shape: [batch_size, seq_len, embed_dim] -> output: [batch_size, seq_len, vocab_size]
    def decoder_projection(self, input_decoder, encoder_output, tgt_kpmask, src_kpmask):
        decoder_output = input_decoder
        for decoder_layer in self.decoder_component:
            decoder_output = decoder_layer(decoder_output, encoder_output, key_padding_mask_tgt = tgt_kpmask, key_padding_mask_src = src_kpmask)
        return self.output_projection(decoder_output)
    
if __name__ == "__main__": 
    
    inputs_id = torch.randint(0, 32000, (16, 512)).to('cuda')
    
    model = Transformer2025().to('cuda')
    
    # Test các thành phần khi phân giải
    embedding_result = model.inference_embedding_layer(inputs_id)    
    print(f"\n---Output embedding layer shape {embedding_result.shape}")
    context_vector = model.inference_encoder_layer(embedding_result, None)
    print(f"\n---Context Vector shape {context_vector.shape}")
    decoder_result = model.inference_decoder_layer(embedding_result, context_vector, None, None)
    print(f"\n---Decoder output shape {decoder_result.shape}")
    logits_result = model.inference_output_projection(decoder_result)
    print(f"\n---Projection output shape {logits_result.shape}")
    decoder_projection = model.decoder_projection(embedding_result, context_vector, None, None)
    print(f"\n---Decoder + Projection output shape {decoder_projection.shape}")
    print()
    
    # from torchviz import make_dot
    # # visualize before backward
    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("linear_cuda_graph", format="png")

    # # compute gradients
    # loss.backward(retain_graph=True)

    # print("Grad X:", x.grad)
    # print("Graph saved as linear_cuda_graph.png")