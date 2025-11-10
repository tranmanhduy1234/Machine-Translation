import sentencepiece as spm

# Tải mô hình .model của bạn
sp = spm.SentencePieceProcessor()
sp.load(r'D:\chuyen_nganh\Machine Translation version2\pre-handle\unigram_32000.model') # Thay bằng đường dẫn file của bạn

# Kiểm tra các ID đặc biệt
print(f"Token <unk>: ID = {sp.unk_id()}")
print(f"Token <s> (BOS): ID = {sp.bos_id()}")
print(f"Token </s> (EOS): ID = {sp.eos_id()}")

# Nếu bạn có định nghĩa token <pad>
if sp.pad_id() != -1: # Mặc định là -1 nếu không được set
    print(f"Token <pad>: ID = {sp.pad_id()}")
else:
    print("Token <pad> không được định nghĩa riêng biệt.")

# Kiểm tra tổng số từ vựng
print(f"Tổng số từ vựng (Vocab size): {sp.get_piece_size()}")