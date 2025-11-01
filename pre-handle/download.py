from datasets import load_dataset

ds = load_dataset("ncduy/mt-en-vi")
train_data = ds["train"]
print(train_data[0])  # Xem 1 mẫu

# Kiểm tra cấu trúc
print("=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

# Tất cả splits
print(f"\n📊 Splits: {list(ds.keys())}")

# Kiểm tra train set
print(f"\n📝 Train set:")
print(f"   - Số lượng: {len(ds['train'])}")
print(f"   - Columns: {ds['train'].column_names}")
print(f"   - Features: {ds['train'].features}")

# Xem 3 samples đầu tiên
print(f"\n🔍 Samples từ train set:")
for i in range(min(3, len(ds['train']))):
    sample = ds['train'][i]
    print(f"\n   Sample {i+1}:")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"      {key}: {value}")
        else:
            print(f"      {key}: {str(value)[:100]}")