from base64 import encode
import torch
import torch.nn as nn
from decoderblock import DecoderBlock
from embedding import Embedding
from encoderblock import EncoderBlock
import sentencepiece as spm

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
        self.embedding = Embedding(self.vocab_size, self.d_model, self.max_length, self.dropout_p) # buộc trọng số bằng cách nào
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
    
# đầu vào yêu cầu của model định dạng [batch_size, seq_len]
model = Transformer2025()
sentences = [
    "Học máy là một lĩnh vực của trí tuệ nhân tạo. 12 312 4213",
    "Tôi tên là Trần Bá Dũng.",
    "Dự án này liên quan đến phát hiện buồn ngủ.",
    "Thử một từ sai: Học máy.",
    "Trong xã hội của chúng ta còn nhiều con người",
    "chịu nhiều thiệt thòi đó họ không đầu hàng số phận không chấp nhận mình sẽ là gánh nặng của gia đình và xã hội họ cố gắng vươn lên trong cuộc sống của chính mình quả là một điều thật sự đáng ngưỡng mộ. Như thầy giáo Nguyễn Ngọc Kí một con người đã chịu rất",
    "Một tên riêng: Phở Thìn Lò Đúc. Traditional",
    "Disadvantaged people, when they were born, were not healthy, but thanks to their extraordinary will and determination to live, they have risen to become successful people who are not inferior to healthy people. Even many young people are healthy and whole, but their efforts in life are not. They let bad habits in life tempt them and then become criminals who commit terrible crimes. These young people are truly a burden to society, they really do not know how to take advantage of the advantages that life and nature have given them, wasting their future youth. Meanwhile, many people are born due to the effects of Agent Orange, or because of nature, they are disadvantaged when they are born, but they always live useful lives, have their own dreams and ambitions"
]
sp = spm.SentencePieceProcessor()
sp.Load(r"D:\chuyen_nganh\Machine Translation version2\pre-handle\unigram_32000.model")

encoded = [sp.EncodeAsIds(s) for s in sentences]

max_len = max(len(seq) for seq in encoded)
pad_id = sp.PieceToId("<pad>") if sp.PieceToId("<pad>") != 0 else 0

padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
x = torch.tensor(padded, dtype=torch.long) # batch_size, seq_len

print("Input shape:", x.shape)
print(model(x, x).shape)
print(f"Model Size: {model.count_parameters()}")