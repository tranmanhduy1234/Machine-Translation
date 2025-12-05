# Quá trình tạo vocab
import sentencepiece as spm 
import numpy as np
import pandas as pd
import time
from pathlib import Path

vocab_sizes = [40000]
results = []

print("=" * 70)
print("TRAINING TOKENIZERS")
print("=" * 70)

for v in vocab_sizes:
    print(f"\nTraining tokenizer with vocab_size={v}...")
    start_time = time.time()
    
    spm.SentencePieceTrainer.train(
        input=r'D:\chuyen_nganh\Machine Translation version2\pre-handle\merged_shuffled.txt',
        model_prefix=f'unigram_{v}',
        vocab_size=v,
        model_type='unigram',
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )
    
    train_time = time.time() - start_time
    print(f"  ✓ Training completed in {train_time:.2f}s")
    
# Đọc file một lần
print("\nLoading text file...")
with open(r'D:\chuyen_nganh\Machine Translation version2\pre-handle\merged_shuffled.txt', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

total_chars = sum(len(s) for s in sentences)
print(f"Total sentences: {len(sentences)}")
print(f"Total characters: {total_chars:,}")

for v in vocab_sizes:
    print(f"\n--- Tokenizer: vocab_size={v} ---")
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=f'unigram_{v}.model')
    
    # 1. Đếm số tokens
    start_time = time.time()
    tokens_per_sentence = [len(sp.encode(s, out_type=str)) for s in sentences]
    encoding_time = time.time() - start_time
    
    total_tokens = sum(tokens_per_sentence)
    avg_tokens = np.mean(tokens_per_sentence)
    median_tokens = np.median(tokens_per_sentence)
    
    # 2. Tính toán các chỉ số hiệu quả
    compression_ratio = total_chars / total_tokens  # Bao nhiêu ký tự trên 1 token
    tokenization_speed = len(sentences) / encoding_time  # Câu/giây
    
    # 3. Kích thước model
    model_size = Path(f'unigram_{v}.model').stat().st_size / (1024 * 1024)  # MB
    
    # 4. Phân phối tokens
    std_tokens = np.std(tokens_per_sentence)
    min_tokens = min(tokens_per_sentence)
    max_tokens = max(tokens_per_sentence)
    
    # Lưu kết quả
    result = {
        'vocab_size': v,
        'total_tokens': total_tokens,
        'avg_tokens_per_sentence': round(avg_tokens, 2),
        'median_tokens_per_sentence': int(median_tokens),
        'std_tokens': round(std_tokens, 2),
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'compression_ratio': round(compression_ratio, 4),
        'tokenization_speed': round(tokenization_speed, 1),
        'encoding_time_sec': round(encoding_time, 2),
        'model_size_mb': round(model_size, 2)
    }
    
    results.append(result)
    
    # In chi tiết
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/sentence: {avg_tokens:.2f}")
    print(f"  Median tokens/sentence: {int(median_tokens)}")
    print(f"  Token std dev: {std_tokens:.2f}")
    print(f"  Token range: [{min_tokens}, {max_tokens}]")
    print(f"  Compression ratio: {compression_ratio:.4f} (chars/token)")
    print(f"  Tokenization speed: {tokenization_speed:.1f} sentences/sec")
    print(f"  Encoding time: {encoding_time:.2f}s")
    print(f"  Model size: {model_size:.2f} MB")

# -----------------------------------------------------------------------------------------------------------------
# Tạo file text phục vụ sinh từ điển từ dataset
# from datasets import load_dataset
# import random

# ds = load_dataset("ncduy/mt-en-vi")

# texts = []

# for split in ["train", "validation", "test"]:
#     if split in ds:
#         dset = ds[split]
#         texts.extend(dset["en"])
#         texts.extend(dset["vi"])

# print(f"Tổng số câu thu được: {len(texts):,}")

# # === 3. Xáo trộn ngẫu nhiên
# random.shuffle(texts)

# # === 4. Ghi ra file
# out_file = "merged_shuffled.txt"
# with open(out_file, "w", encoding="utf-8") as f:
#     for line in texts:
#         if line.strip():
#             f.write(line.strip() + "\n")

# print(f"Đã ghi xong: {out_file}")