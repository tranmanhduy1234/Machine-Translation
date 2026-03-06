import sentencepiece as spm
from typing import List, Union
import numpy as np

class Tokenizer2025:
    def __init__(self, model_spm_path="", legacy=False):
        self.model_spm_path = model_spm_path
        self.sp = spm.SentencePieceProcessor(model_file=model_spm_path)
        
        # SentencePiece không cho phép add_special_tokens động.
        # Ta lấy ID từ file model, nếu không có sẽ trả về -1.
        self.pad_token = "<pad>"
        self.pad_id = self.sp.piece_to_id(self.pad_token)

    def print_infor_vocab(self):
        # Fix: Sử dụng id_to_piece kết hợp với các hàm lấy ID
        print(f"Kích thước từ điển: {self.sp.get_piece_size()}")
        print(f"UNK token: {self.sp.id_to_piece(self.sp.unk_id())} (ID: {self.sp.unk_id()})")
        print(f"BOS token: {self.sp.id_to_piece(self.sp.bos_id())} (ID: {self.sp.bos_id()})")
        print(f"EOS token: {self.sp.id_to_piece(self.sp.eos_id())} (ID: {self.sp.eos_id()})")
        
        pad_display = self.pad_id if self.pad_id != -1 else "N/A (Not in vocab)"
        print(f"PAD token: {self.pad_token} (ID: {pad_display})")
        
    def encode(self, texts: List[str]):
        all_ids = []
        all_pieces = []
        
        # Lấy ID đặc biệt
        bos_id = self.sp.bos_id()
        eos_id = self.sp.eos_id()

        for text in texts:
            # Encode text thô
            ids = self.sp.encode(text, out_type=int)
            # Ghép BOS và EOS thủ công như logic cũ của bạn
            full_ids = [bos_id] + ids + [eos_id]
            
            all_ids.append(np.array(full_ids))
            # Chuyển list ID sang list các "miếng" token để debug
            all_pieces.append([self.sp.id_to_piece(idx) for idx in full_ids])

        return all_ids, all_pieces
    
    def decode(self, ids_batch: List[Union[List[int], np.ndarray]], skip_special_tokens: bool = True):
        decoded_texts = []
        for ids in ids_batch:
            # Chuyển về list int nếu là numpy array
            curr_ids = ids.tolist() if isinstance(ids, np.ndarray) else ids
            
            if skip_special_tokens:
                # SentencePiece có tham số xử lý trực tiếp các ID đặc biệt cơ bản
                # Tuy nhiên để chắc chắn khớp logic Llama, ta có thể lọc thủ công:
                special_ids = {self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id(), self.pad_id}
                curr_ids = [idx for idx in curr_ids if idx not in special_ids]
            
            decoded_texts.append(self.sp.decode(curr_ids))
        return decoded_texts
    
    # Getters (đã fix tên hàm)
    def get_unk_token(self): return self.sp.unk_id(), self.sp.id_to_piece(self.sp.unk_id())
    def get_bos_token(self): return self.sp.bos_id(), self.sp.id_to_piece(self.sp.bos_id())
    def get_eos_token(self): return self.sp.eos_id(), self.sp.id_to_piece(self.sp.eos_id())
    def get_pad_token(self): return self.pad_id, self.pad_token

if __name__ == "__main__":
    # Thay đường dẫn chuẩn của bạn ở đây
    MODEL_PATH = r'D:\chuyen_nganh\Machine-Translation\source\tokenizer\unigram_40000.model'
    
    try:
        tokenizer = Tokenizer2025(model_spm_path=MODEL_PATH)
        tokenizer.print_infor_vocab()
        
        batch_texts = ["Trần Đỗ Mạnh Duy. Ngày mai rồi sẽ khác", "Xin chào thế giới"]
        encoded_ids, token_pieces = tokenizer.encode(texts=batch_texts)
        
        print(f"\nCâu 1 Pieces: {token_pieces[0]}")
        print(f"Câu 1 IDs: {encoded_ids[0]}")
        
        decoded = tokenizer.decode([encoded_ids[0]])
        print(f"Decoded: {decoded}")
    except Exception as e:
        print(f"Lỗi khi thực thi: {e}")