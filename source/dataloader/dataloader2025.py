from datasets import load_dataset
from torch.utils.data import DataLoader
from source.tokenizer.tokenizer2025 import Tokenizer2025
from config import * 
import numpy as np
import torch
# steps: 115607
class TranslationDataloader:
    def __init__(self, path_tsv, tokenizer: Tokenizer2025):
        self.path_tsv = path_tsv
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            'csv',
            data_files={'train': self.path_tsv},
            sep='\t',
            streaming=True
        )
        
    def getDataloader(self, batch_size = -1):
        if batch_size == -1:
            batch_size = BATCH_SIZE
        streamed_dataset = self.dataset["train"]
        return DataLoader(
            streamed_dataset, 
            batch_size=batch_size, 
            collate_fn=self.collate_fn,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
            drop_last=DROPLAST,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
        )
    
    def collate_fn(self, batch):
        # batch dạng [{'en': 'Support quickly.', 'vi': 'Hỗ trợ nhanh chóng.'}, {'en': 'Time Consumption', 'vi': 'Sự tiêu thụ thời gian'}...]
        batch_size = len(batch)
        ens = [pair['en'] for pair in batch]
        vis = [pair['vi'] for pair in batch]
        ens_encoded_ids, ens_encoded_pieces = self.tokenizer.encode(texts=ens)
        vis_encoded_ids, vis_encoded_pieces = self.tokenizer.encode(texts=vis)
        
        max_length_src = max(map(len, ens_encoded_ids))
        max_length_tgt = max(map(len, vis_encoded_ids))
        padding_token_id = self.tokenizer.get_pad_token()[0]
        
        ens_encoded_ids_padded = np.full((batch_size, max_length_src), 
                                         fill_value=padding_token_id, 
                                         dtype=np.int64)
        for i, arr in enumerate(ens_encoded_ids):
            ens_encoded_ids_padded[i, :len(arr)] = arr

        vis_encoded_ids_padded = np.full((batch_size, max_length_tgt), 
                                         fill_value=padding_token_id, 
                                         dtype=np.int64)
        for i, arr in enumerate(vis_encoded_ids):
            vis_encoded_ids_padded[i, :len(arr)] = arr
            
        ens_keypaddingmask = (ens_encoded_ids_padded == padding_token_id)
        vis_keypaddingmask = (vis_encoded_ids_padded == padding_token_id)
        
        # chuyển đổi thành đầu ra
        vis_source = torch.from_numpy(vis_encoded_ids_padded)
        vis_target = torch.roll(vis_source, shifts=-1, dims=-1)
        vis_target[:, -1] = padding_token_id
        return {
            "en_ids": torch.from_numpy(ens_encoded_ids_padded),
            "vi_ids_src": vis_source,
            "vi_ids_tgt": vis_target,
            "en_mask": torch.from_numpy(ens_keypaddingmask),
            "vi_mask": torch.from_numpy(vis_keypaddingmask),
            "en_text": ens,
            "vi_text": vis
        }
    
    def print_config_dataloader(self):
        print(f"Config DataLoader:")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Num workers: {NUM_WORKERS}")
        print(f"  - Pin memory: {PIN_MEMORY}")
        print(f"  - Drop last: {DROPLAST}")
        print(f"  - Persistent workers: {PERSISTENT_WORKERS}")
        print()
if __name__=="__main__":
    data = TranslationDataloader(
        path_tsv=r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_test.tsv",
        tokenizer=Tokenizer2025(MODEL_SPM_PATH)
    )
    datatrain = data.getDataloader()
    count = 0
    for _ in datatrain:
        count += 1
    print(count)
    for i, batch in enumerate(datatrain):
        print(batch)
        print(f"Batch {i+1}:")
        print(f"  en_ids shape: {batch['en_ids'].shape}")
        print(f"  vi_ids_src shape: {batch['vi_ids_src'].shape}")
        print(f"  vi_ids_tgt shape: {batch['vi_ids_tgt'].shape}")
        
        print(f"  en_mask shape: {batch['en_mask'].shape}")
        print(f"  vi_mask shape: {batch['vi_mask'].shape}")
        print()
        print(f"Batch data:")
        print(f"  en_ids dtype: {batch['en_ids'].dtype}")
        print(f"  en_mask dtype: {batch['en_mask'].dtype}")
        print()
        print(f"Sample en_text: {batch['en_text'][:2]}")
        print(f"Sample vi_text: {batch['vi_text'][:2]}")
        break