import sentencepiece as spm 
import numpy as np
import pandas as pd
import time
from pathlib import Path

# 1. Huấn luyện các tokenizer với vocab_size khác nhau
vocab_sizes = [8000, 16000, 32000, 64000]
results = []

print("=" * 70)
print("TRAINING TOKENIZERS")
print("=" * 70)

for v in vocab_sizes:
    print(f"\nTraining tokenizer with vocab_size={v}...")
    start_time = time.time()
    
    spm.SentencePieceTrainer.train(
        input='merged_shuffled.txt',
        model_prefix=f'unigram_{v}',
        vocab_size=v,
        model_type='unigram'
    )
    
    train_time = time.time() - start_time
    print(f"  ✓ Training completed in {train_time:.2f}s")

# 2. Đánh giá từng tokenizer
print("\n" + "=" * 70)
print("EVALUATING TOKENIZERS")
print("=" * 70)

# Đọc file một lần
print("\nLoading text file...")
with open('merged_shuffled.txt', encoding='utf-8') as f:
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

# 3. Tạo bảng tổng hợp
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

# 4. Tính toán so sánh với baseline (vocab_size=16000)
print("\n" + "=" * 70)
print("COMPARISON WITH BASELINE (vocab_size=16000)")
print("=" * 70)

baseline_idx = 1  # vocab_size=16000
baseline_tokens = results[baseline_idx]['total_tokens']
baseline_speed = results[baseline_idx]['tokenization_speed']

print(f"\nBaseline: vocab_size=16000 ({baseline_tokens:,} tokens)")

for i, v in enumerate(vocab_sizes):
    if i == baseline_idx:
        continue
    
    token_change = ((results[i]['total_tokens'] - baseline_tokens) / baseline_tokens) * 100
    speed_change = ((results[i]['tokenization_speed'] - baseline_speed) / baseline_speed) * 100
    
    print(f"\nvocab_size={v}:")
    print(f"  Token count change: {token_change:+.1f}%")
    print(f"  Speed change: {speed_change:+.1f}%")

# 5. Đưa ra khuyến nghị
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

best_compression = max(results, key=lambda x: x['compression_ratio'])
best_speed = max(results, key=lambda x: x['tokenization_speed'])
best_balanced = results[1]  # vocab_size=16000 thường là cân bằng

print(f"\n✓ Best compression ratio: vocab_size={best_compression['vocab_size']} ({best_compression['compression_ratio']:.4f})")
print(f"✓ Fastest tokenization: vocab_size={best_speed['vocab_size']} ({best_speed['tokenization_speed']:.1f} sent/sec)")
print(f"✓ Balanced choice: vocab_size={best_balanced['vocab_size']}")

# 6. Lưu kết quả
df.to_csv('tokenizer_efficiency_results.csv', index=False)
print("\n✓ Results saved to 'tokenizer_efficiency_results.csv'")

# Tạo file text phục vụ sinh từ điển từ dataset
 """
from datasets import load_dataset
import random

ds = load_dataset("ncduy/mt-en-vi")

texts = []

for split in ["train", "validation", "test"]:
    if split in ds:
        dset = ds[split]
        texts.extend(dset["en"])
        texts.extend(dset["vi"])

print(f"Tổng số câu thu được: {len(texts):,}")

# === 3. Xáo trộn ngẫu nhiên
random.shuffle(texts)

# === 4. Ghi ra file
out_file = "merged_shuffled.txt"
with open(out_file, "w", encoding="utf-8") as f:
    for line in texts:
        if line.strip():
            f.write(line.strip() + "\n")

print(f"Đã ghi xong: {out_file}")
"""