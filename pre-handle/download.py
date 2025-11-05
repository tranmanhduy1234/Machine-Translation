# from datasets import load_dataset

# ds = load_dataset("ncduy/mt-en-vi")
# train_data = ds["train"]
# print(train_data[0])  # Xem 1 mẫu

# # Kiểm tra cấu trúc
# print("=" * 60)
# print("DATASET STRUCTURE")
# print("=" * 60)

# # Tất cả splits
# print(f"\n📊 Splits: {list(ds.keys())}")

# # Kiểm tra train set
# print(f"\n📝 Train set:")
# print(f"   - Số lượng: {len(ds['train'])}")
# print(f"   - Columns: {ds['train'].column_names}")
# print(f"   - Features: {ds['train'].features}")

# # Xem 3 samples đầu tiên
# print(f"\n🔍 Samples từ train set:")
# for i in range(min(3, len(ds['train']))):
#     sample = ds['train'][i]
#     print(f"\n   Sample {i+1}:")
#     for key, value in sample.items():
#         if isinstance(value, dict):
#             print(f"      {key}: {value}")
#         else:
#             print(f"      {key}: {str(value)[:100]}")

# kiểm tra nguồn
# from datasets import load_dataset
# from collections import Counter

# # Tải dataset
# ds = load_dataset("ncduy/mt-en-vi", split="train")

# # Đếm tần suất 'source'
# counter = Counter(ds["source"])

# # Tổng số mẫu
# total = sum(counter.values())

# # In top nguồn phổ biến
# print(f"{'Source':<20} | {'Count':>10} | {'Percent':>8}")
# print("-" * 45)
# for src, cnt in counter.most_common():
#     print(f"{src:<20} | {cnt:>10,} | {cnt/total*100:>7.2f}%")

# Nếu muốn lưu ra CSV
# import pandas as pd
# df = pd.DataFrame(counter.items(), columns=["source", "count"])
# df["percent"] = df["count"] / total * 100
# df.to_csv("source_stats.csv", index=False)

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