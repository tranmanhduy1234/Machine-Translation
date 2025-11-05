# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('tokenizer_efficiency_results.csv')

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # Biểu đồ 1: Total Tokens
# axes[0, 0].plot(df['vocab_size'], df['total_tokens']/1e6, marker='o', linewidth=2, markersize=8)
# axes[0, 0].set_title('Total Tokens', fontsize=12, fontweight='bold')
# axes[0, 0].set_xlabel('Vocab Size')
# axes[0, 0].set_ylabel('Tokens (M)')
# axes[0, 0].grid(True, alpha=0.3)

# # Biểu đồ 2: Compression Ratio
# axes[0, 1].bar(df['vocab_size'].astype(str), df['compression_ratio'], color='skyblue', edgecolor='black')
# axes[0, 1].set_title('Compression Ratio (chars/token)', fontsize=12, fontweight='bold')
# axes[0, 1].set_ylabel('Ratio')
# axes[0, 1].grid(axis='y', alpha=0.3)

# # Biểu đồ 3: Speed
# axes[1, 0].bar(df['vocab_size'].astype(str), df['tokenization_speed']/1000, color='lightcoral', edgecolor='black')
# axes[1, 0].set_title('Tokenization Speed', fontsize=12, fontweight='bold')
# axes[1, 0].set_ylabel('K sentences/sec')
# axes[1, 0].grid(axis='y', alpha=0.3)

# # Biểu đồ 4: Model Size
# axes[1, 1].bar(df['vocab_size'].astype(str), df['model_size_mb'], color='lightgreen', edgecolor='black')
# axes[1, 1].set_title('Model Size', fontsize=12, fontweight='bold')
# axes[1, 1].set_ylabel('Size (MB)')
# axes[1, 1].grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.savefig('tokenizer_comparison.png', dpi=300, bbox_inches='tight')
# print("Chart saved: tokenizer_comparison.png")
# plt.show()

# print("\n" + df.to_string())
# ------------------------------------------------------------------------------------------------------------
# import sentencepiece as spm
# from datasets import load_dataset
# import numpy as np, time

# sp = spm.SentencePieceProcessor(model_file='unigram_32000.model')

# # Ví dụ dùng tập dữ liệu nhỏ để kiểm tra
# texts = ["This is an example sentence.", "Machine translation is fun.", "Xin chào, tôi là Duy."]
# start = time.time()
# encoded = [sp.encode(text, out_type=str) for text in texts]
# elapsed = time.time() - start

# num_tokens = [len(e) for e in encoded]
# avg_tokens = np.mean(num_tokens)
# compression_ratio = sum(len(" ".join(e)) for e in encoded) / sum(len(t) for t in texts)

# print(f"Average tokens per sentence: {avg_tokens:.2f}")
# print(f"Compression ratio: {compression_ratio:.3f}")
# print(f"Tokenization speed: {len(texts)/elapsed:.2f} sentences/sec")

# ------------------------------------------------------------------------------------------------------------

# import sentencepiece as spm
# sp = spm.SentencePieceProcessor(model_file='unigram_32000.model')

# samples = [
#   "Xin chào.",
#   "Tôi đi học.",
#   "Machine translation",
#   "Hôm nay trời mưa to.",
#   "OK"
# ]

# for s in samples:
#     print(s)
#     print(sp.encode(s, out_type=str))
#     print()

# ------------------------------------------------------------------------------------------------------------
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r"D:\chuyen_nganh\Machine Translation version2\pre-handle\unigram_32000.model")

print(f"Vocab size: {sp.GetPieceSize()}")

sentences = [
    "Học máy là một lĩnh vực của trí tuệ nhân tạo.",
    "Tôi tên là Trần Bá Dũng.",
    "Dự án này liên quan đến phát hiện buồn ngủ.",
    "Thử một từ sai: Hcoj máy.",
    "Trong xã hội của chúng ta còn nhiều con người",
    "chịu nhiều thiệt thòi đó họ không đầu hàng số phận không chấp nhận mình sẽ là gánh nặng của gia đình và xã hội họ cố gắng vươn lên trong cuộc sống của chính mình quả là một điều thật sự đáng ngưỡng mộ. Như thầy giáo Nguyễn Ngọc Kí một con người đã chịu rất",
    "Một tên riêng: Phở Thìn Lò Đúc. Traditional",
    "Disadvantaged people, when they were born, were not healthy, but thanks to their extraordinary will and determination to live, they have risen to become successful people who are not inferior to healthy people. Even many young people are healthy and whole, but their efforts in life are not. They let bad habits in life tempt them and then become criminals who commit terrible crimes. These young people are truly a burden to society, they really do not know how to take advantage of the advantages that life and nature have given them, wasting their future youth. Meanwhile, many people are born due to the effects of Agent Orange, or because of nature, they are disadvantaged when they are born, but they always live useful lives, have their own dreams and ambitions"
]

print("\n--- Kết quả Tokenization ---")
for s in sentences:
    pieces = sp.EncodeAsPieces(s)
    ids = sp.EncodeAsIds(s)
    
    print(f"\nCâu gốc: {s}")
    print(f"Tokens: {pieces}")
# # ------------------------------------------------------------------------------------------------------------