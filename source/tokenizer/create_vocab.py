import sentencepiece as spm
import numpy as np

vocab_size = 40000
print("="*50)
print(f"Training tokenizers with vocab size {vocab_size} token")
print("="*50)
input_txt = r"D:\chuyen_nganh\Machine Translation version2\source\tokenizer\tokenization.txt"
spm.SentencePieceTrainer.train(
    input=input_txt, # file txt
    model_prefix=f"unigram_{vocab_size}",
    vocab_size=vocab_size,
    model_type='unigram',
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3
)
print("Traning hoàn tất")
print("\nLoad file text kiểm thử...")
with open(input_txt, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]
    
total_chars = sum(len(s) for s in sentences)
print(f"Total sentences: {len(sentences)}")
print(f"Total characters: {total_chars:,}")

print("Load model...")
sp = spm.SentencePieceProcessor(model_file=f"unigram_{vocab_size}.model")
tokens_per_sentence = [len(sp.encode(s, out_type=str)) for s in sentences]

print("Load model thành công, bắt đầu tính toán")
total_tokens = sum(tokens_per_sentence)
avg_tokens = np.mean(tokens_per_sentence)
median_tokens = np.median(tokens_per_sentence)

compression_ratio = total_chars / total_tokens
std_tokens = np.std(tokens_per_sentence)
min_tokens = min(tokens_per_sentence)
max_tokens = max(tokens_per_sentence)

print(f"  Total tokens: {total_tokens:,}")
print(f"  Avg tokens/sentence: {avg_tokens:.2f}")
print(f"  Median tokens/sentence: {int(median_tokens)}")
print(f"  Token std dev: {std_tokens:.2f}")
print(f"  Token range: [{min_tokens}, {max_tokens}]")
print(f"  Compression ratio: {compression_ratio:.4f} (chars/token)")