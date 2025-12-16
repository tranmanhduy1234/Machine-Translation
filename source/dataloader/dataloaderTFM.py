"""
    BẢN DEMO 
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List

class TranslationDataset(Dataset):
    """Dataset for English-Vietnamese translation pairs"""
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        """
        Args:
            csv_path: Path to CSV file with 'en' and 'vi' columns
            tokenizer: Tokenizer (from transformers library)
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate columns exist
        if 'en' not in self.df.columns or 'vi' not in self.df.columns:
            raise ValueError("CSV must contain 'en' and 'vi' columns")
        
        # Remove rows with missing values
        self.df = self.df.dropna(subset=['en', 'vi'])
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        en_text = str(self.df.iloc[idx]['en']).strip()
        vi_text = str(self.df.iloc[idx]['vi']).strip()
        
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
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for translation pairs
    
    Args:
        csv_path: Path to CSV file
        tokenizer: Tokenizer from transformers
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader instance
    """
    dataset = TranslationDataset(csv_path, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    import os
    root = r"D:\chuyen_nganh\Dataset MT\ccmatrix"
    file_paths = os.listdir(root)
    total_step = 0
    for file in file_paths:
        link = os.path.join(root, file)
        if os.path.exists(link):
            dataloader = create_dataloader(
                csv_path=link,
                tokenizer=tokenizer,
                batch_size=256,
                max_length=512,
                shuffle=True,
                num_workers=0
            )
            total_step += len(dataloader)
            print(len(dataloader))
            
    exit(0)
    # Create dataloader
    dataloader = create_dataloader(
        csv_path=r"D:\chuyen_nganh\Dataset MT\vietai_mtet\data_part_01.csv",
        tokenizer=tokenizer,
        batch_size=256,
        max_length=512,
        shuffle=True,
        num_workers=0
    )
    print(len(dataloader))