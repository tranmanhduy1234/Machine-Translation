# ============================================== thống kê độ dài
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# Tải tập train
ds = load_dataset("ncduy/mt-en-vi", split="train")

# Tính độ dài ký tự & từ
en_lengths_char = [len(x) for x in ds["en"]]
vi_lengths_char = [len(x) for x in ds["vi"]]
en_lengths_word = [len(x.split()) for x in ds["en"]]
vi_lengths_word = [len(x.split()) for x in ds["vi"]]

def describe(name, lengths):
    print(f"--- {name} ---")
    print(f"  Trung bình : {np.mean(lengths):.2f}")
    print(f"  Trung vị   : {np.median(lengths):.2f}")
    print(f"  Độ lệch chuẩn : {np.std(lengths):.2f}")
    print(f"  Min - Max  : {np.min(lengths)} - {np.max(lengths)}")
    print()

# In thống kê
describe("EN (từ)", en_lengths_word)
describe("VI (từ)", vi_lengths_word)

# Vẽ histogram phân bố độ dài
plt.figure(figsize=(10,5))
plt.hist(en_lengths_word, bins=80, alpha=0.6, label="English")
plt.hist(vi_lengths_word, bins=80, alpha=0.6, label="Vietnamese")
plt.xlabel("Độ dài câu (số từ)")
plt.ylabel("Số lượng mẫu")
plt.legend()
plt.title("Phân bố độ dài câu Anh - Việt")
plt.show()