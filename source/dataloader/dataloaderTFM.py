"""
    PHIÊN BẢN TỐI ƯU CHO FILE TSV LỚN
    Hoạt động trên dataset cho bài toán machine translate
    - Tổng số lượng cặp câu: 30199260 
    - Tổng số lượng token: 1872206530 (Bộ token cá nhân)
    - Tổng số lượng batch: 117966 (Batch: 256)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import warnings

class TranslationDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Đang tải dữ liệu từ {csv_path}...")
        
        # Đọc file TSV với tối ưu bộ nhớ
        # Chỉ đọc 2 cột cần thiết và sử dụng dtype để tiết kiệm RAM
        try:
            self.df = pd.read_csv(
                csv_path, 
                sep='\t',
                usecols=['en', 'vi'],  # Chỉ đọc 2 cột cần thiết
                dtype=str,  # Đảm bảo đọc dưới dạng string
                na_filter=True,  # Lọc NA
                encoding='utf-8',
                engine='c'  # Engine C nhanh hơn Python engine
            )
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            raise
        
        # Kiểm tra columns
        if 'en' not in self.df.columns or 'vi' not in self.df.columns:
            raise ValueError("CSV phải chứa cột 'en' và 'vi'")
        
        # Xóa các dòng thiếu dữ liệu
        original_len = len(self.df)
        self.df = self.df.dropna(subset=['en', 'vi'])
        
        # Xóa các dòng trống
        self.df = self.df[(self.df['en'].str.strip() != '') & (self.df['vi'].str.strip() != '')]
        
        # Reset index sau khi drop
        self.df.reset_index(drop=True, inplace=True)
        
        removed = original_len - len(self.df)
        if removed > 0:
            print(f"Đã loại bỏ {removed} dòng thiếu dữ liệu")
        
        print(f"Đã tải {len(self.df):,} mẫu dữ liệu thành công")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        en_text = self.df.iloc[idx]['en'].strip()
        vi_text = self.df.iloc[idx]['vi'].strip()
        
        # Tokenize English text
        en_encoded = self.tokenizer(
            en_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize Vietnamese text
        vi_encoded = self.tokenizer(
            vi_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'en_input_ids': en_encoded['input_ids'].squeeze(0),
            'en_attention_mask': en_encoded['attention_mask'].squeeze(0),
            'vi_input_ids': vi_encoded['input_ids'].squeeze(0),
            'vi_attention_mask': vi_encoded['attention_mask'].squeeze(0),
        }

def create_dataloader(
    csv_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,  # Sửa default
    pin_memory: bool = True  # Thêm pin_memory cho tốc độ
) -> DataLoader:
    """
    Tạo DataLoader cho dữ liệu dịch thuật
    
    Args:
        csv_path: Đường dẫn đến file TSV
        tokenizer: Tokenizer để mã hóa văn bản
        batch_size: Kích thước batch
        max_length: Độ dài tối đa của sequence
        shuffle: Có xáo trộn dữ liệu không
        num_workers: Số worker để load dữ liệu song song
        persistent_workers: Giữ workers sống giữa các epoch
        pin_memory: Pin memory cho GPU (nhanh hơn khi train trên GPU)
    """
    dataset = TranslationDataset(csv_path, tokenizer, max_length)
    
    # Chỉ dùng persistent_workers khi num_workers > 0
    use_persistent = persistent_workers and num_workers > 0
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        pin_memory=pin_memory and torch.cuda.is_available(),  # Chỉ pin khi có GPU
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch để tăng tốc
    )
    
    print(f"DataLoader được tạo với {len(dataloader):,} batches")
    return dataloader

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("Đang tải tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    
    # Đường dẫn file của bạn
    csv_path = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\first1000.tsv"
    
    print("\nĐang tạo dataloader...")
    dataloader = create_dataloader(
        csv_path=csv_path,
        tokenizer=tokenizer,
        batch_size=256,
        max_length=512,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    
    print(f"\nSố lượng batches: {len(dataloader):,}")
    print(f"Tổng số mẫu: {len(dataloader.dataset):,}")
    
    # Test lấy 1 batch
    print("\nĐang lấy batch đầu tiên để test...")
    batch = next(iter(dataloader))
    
    print(f"\nThông tin batch:")
    print(f"  - en_input_ids shape: {batch['en_input_ids'].shape}")
    print(f"  - en_attention_mask shape: {batch['en_attention_mask'].shape}")
    print(f"  - vi_input_ids shape: {batch['vi_input_ids'].shape}")
    print(f"  - vi_attention_mask shape: {batch['vi_attention_mask'].shape}")
    
    # Giải mã 1 ví dụ để kiểm tra
    print("\nVí dụ mẫu đầu tiên:")
    en_sample = tokenizer.decode(batch['en_input_ids'][0], skip_special_tokens=True)
    vi_sample = tokenizer.decode(batch['vi_input_ids'][0], skip_special_tokens=True)
    print(f"  EN: {en_sample[:100]}...")
    print(f"  VI: {vi_sample[:100]}...")