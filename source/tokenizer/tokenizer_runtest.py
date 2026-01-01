import sentencepiece as spm
from transformers import LlamaTokenizerFast
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

model_spm_path = r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model'

class Tokenizer2025:
    def __init__(self, model_spm_path="", legacy=False):
        self.model_spm_path = model_spm_path
        self.legacy = legacy
        self.tokenizer = LlamaTokenizerFast(vocab_file=self.model_spm_path, legacy=False)
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
    def print_infor_vocab(self):
        print(f"Kích thước từ điển: {len(self.tokenizer)}")
        print(f"UNK token: {self.tokenizer.unk_token} (ID: {self.tokenizer.unk_token_id})")
        print(f"BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
        print(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        
    def encode(self, text, max_length):
        text = "".join([self.tokenizer.bos_token, text, self.tokenizer.eos_token])
        print(text)
        encoded = self.tokenizer.encode(
            text=text,
            padding='max_length',
            max_length=max_length,
            add_special_tokens=False,
            padding_side='right',
            truncation=True
        )
        return encoded, self.tokenizer.convert_ids_to_tokens(encoded)
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def get_unk_token(self):
        return self.tokenizer.unk_token_id, self.tokenizer.unk_token
    def get_bos_token(self):
        return self.tokenizer.bos_token_id, self.tokenizer.bos_token
    def get_eos_token(self):
        return self.tokenizer.eos_token_id, self.tokenizer.eos_token 
    def get_pad_token(self):
        return self.tokenizer.pad_token_id, self.tokenizer.pad_token
            
if __name__=="__main__":
    tokenizer = Tokenizer2025(model_spm_path=model_spm_path, legacy=False)
    tokenizer.print_infor_vocab()
    text = "Trần Đỗ Mạnh Duy - Ngày mai rồi sẽ khác"
    encoded, token_pieces = tokenizer.encode(text=text, max_length=20)
    print(f"Các mảnh token sau encode\n{token_pieces}")
    print(encoded)
    print(tokenizer.decode(encoded))
    print(tokenizer.get_eos_token())