def preprocess_function(examples):
    inputs = examples['en']
    targets = examples['vi']
    return inputs, targets

import torch
from datasets import load_dataset
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
# === 1. TẢI DATASET ===
print("Tải dataset...")
try:
    ds = load_dataset("ncduy/mt-en-vi")
    print("Tải xong")
    print(f"   - Train samples: {len(ds['train'])}")
    print(f"   - Validation samples: {len(ds['validation'])}")
    print(f"   - Test samples: {len(ds['test'])}")
    
    # Xem sample
    sample = ds['train'][0]
    print(f"   - Sample: EN: {sample['en'][:50]}...")
    print(f"             VI: {sample['vi'][:50]}...")
except Exception as e:
    print(f"Lỗi tải dataset: {e}")
    exit(1)

# === 2. TẢI MODEL ===
print("\nĐang tải model Marian (EN->VI)...")
try:
    model_name = "Helsinki-NLP/Opus-MT-en-vi"
    print(f"   Model: {model_name}")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model tải thành công!")
    print(f"   - Device: {device}")
    print(f"   - Model params: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"Lỗi tải model: {e}")
    exit(1)

# === 3. TIỀN XỬ LÝ DỮ LIỆU ===
print("\nĐang tiền xử lý dữ liệu...")
def preprocess_function(examples):
    inputs = examples['en']
    targets = examples['vi']
    
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    return model_inputs
try:
    # Xử lý train set
    tokenized_train = ds['train'].map(
        preprocess_function,
        batched=True,
        batch_size=16,
        remove_columns=['en', 'vi', 'source'],
        desc="Processing train data"
    )
    print(f"Train data processed: {len(tokenized_train)} samples")
    # Xử lý validation set
    if 'validation' in ds:
        tokenized_val = ds['validation'].map(
            preprocess_function,
            batched=True,
            batch_size=16,
            remove_columns=['en', 'vi', 'source'],
            desc="Processing validation data"
        )
        print(f"Validation data processed: {len(tokenized_val)} samples")
    else:
        # Nếu không có validation, split train data
        split_data = tokenized_train.train_test_split(test_size=0.1)
        tokenized_train = split_data['train']
        tokenized_val = split_data['test']
        print(f"Chia train/val: {len(tokenized_train)}/{len(tokenized_val)}")
except Exception as e:
    print(f"Lỗi tiền xử lý: {e}")
    exit(1)

# === 4. CẤU HÌNH HUẤN LUYỆN ===
print("\nCấu hình huấn luyện...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt_model_checkpoint",
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_steps=100,
    predict_with_generate=True,
    seed=42,
)
print("Cấu hình hoàn tất")
# === 5. TẠO TRAINER ===
print("\nTạo Trainer...")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Trainer tạo thành công")

# === 6. BẮT ĐẦU HUẤN LUYỆN ===
print("\nBẮT ĐẦU HUẤN LUYỆN...")
print(f"   - Total train steps: ~{len(tokenized_train) // 8 * 2}")
print("=" * 60)

try:
    trainer.train()
    print("=" * 60)
    print("HUẤN LUYỆN HOÀN TẤT!")
except Exception as e:
    print(f"Lỗi huấn luyện: {e}")
    exit(1)

# === 7. LƯU MODEL ===
print("\nLưu model...")
try:
    model.save_pretrained("./my_mt_model")
    tokenizer.save_pretrained("./my_mt_model")
    print("Model lưu tại: ./my_mt_model")
except Exception as e:
    print(f"Lỗi lưu model: {e}")
    exit(1)

# === 8. KIỂM TRA MÔ HÌNH ===
print("\nKiểm tra mô hình...")
try:
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50)
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"   EN: {test_text}")
    print(f"   VI: {translated}")
    print("Model hoạt động bình thường!")
except Exception as e:
    print(f"Lỗi kiểm tra: {e}")