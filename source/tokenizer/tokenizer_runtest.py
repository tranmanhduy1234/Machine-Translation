import sentencepiece as spm

# Tải mô hình .model của bạn
sp = spm.SentencePieceProcessor()
sp.load(r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model') # Thay bằng đường dẫn file của bạn

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

print("\n--- Kết quả Tokenization ---")
s = """
Mẹ tôi là người mà tôi ngưỡng mộ nhất. Mẹ đã cống hiến nhiều thời gian và sức lực vào việc dạy dỗ tôi và hai anh trai tôi. Mặc dù làm việc vất vả nhưng bà ấy đã luôn dành thời gian để dạy chúng tôi nhiều điều bổ ích mà cần thiết và quan trọng trong cuộc sống sau này của chúng tôi. Hơn nữa, mẹ là một tấm gương cho tôi noi theo. Mẹ luôn cố gắng sống hòa thuận với những người hàng xóm bên cạnh và giúp đỡ mọi người khi họ gặp khó khăn cho nên hầu hết mọi người tôn trọng và yêu quý bà ấy. Tôi ngưỡng mộ và kính trọng mẹ tôi không chỉ bởi vì bà nuôi dưỡng tôi tốt mà mẹ còn bên tôi và đưa ra sự giúp đỡ nếu cần thiết. Ví dụ như khi tôi gập những khó khăn thì mẹ sẽ đưa ra những lời khuyên quý giá giúp tôi giải quyết những vấn đề đó. Mẹ có ảnh hưởng lớn tới tôi và tôi hi vọng rằng tôi sẽ thừa hưởng được một số nét tính cách của mẹ.
"""

pieces = sp.EncodeAsPieces(s)
ids = sp.EncodeAsIds(s)
# print(f"Tokens: {pieces}")
print((ids))
print(len(ids))
