"""
THÀNH PHẦN ĐÁNH GIÁ TỐC ĐỘ SO VỚI MODEL MarianMT
"""
import torch
import time
from transformers import MarianMTModel, MarianTokenizer
from source.build_model.model import Transformer2025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _sync_if_cuda():
    if device.type == "cuda":
        torch.cuda.synchronize()

# ======== LOAD MODELS ========
model_name = "Helsinki-NLP/Opus-MT-en-vi"
print(f"Model: {model_name}")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model_marian = MarianMTModel.from_pretrained(model_name).to(device).eval()

model_custom = Transformer2025().to(device).eval()

# ======== INPUT ========
text = "Machine learning models are changing how we solve real-world problems."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
x = inputs["input_ids"]  # [batch, seq_len]

# ======== WARM-UP ========
for _ in range(3):
    with torch.no_grad():
        # Marian forward expects input_ids and decoder_input_ids
        _ = model_marian(input_ids=x, decoder_input_ids=x)
        # Custom transformer expects (src, tgt)
        _ = model_custom(x, x)

# ======== BENCHMARK ========
def benchmark_model(name, fn, *args, **kwargs):
    _sync_if_cuda()
    start = time.time()
    with torch.no_grad():
        _ = fn(*args, **kwargs)
    _sync_if_cuda()
    end = time.time()
    print(f"{name}: {(end - start) * 1000:.2f} ms")

print("\n===== BENCHMARK =====")
benchmark_model("MarianMT (core)", model_marian, input_ids=x, decoder_input_ids=x)
benchmark_model("Transformer2025 (custom)", model_custom, x, x)

# ======== PARAM COUNT ========
params_marian = sum(p.numel() for p in model_marian.parameters())
params_custom = sum(p.numel() for p in model_custom.parameters())
print(f"\nMarianMT params: {params_marian:,}")
print(f"Transformer2025 params: {model_custom.count_parameters():,}")