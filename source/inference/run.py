"""
NƠI CHẠY MODEL - THỰC NGHIỆM KẾT QUẢ
"""
import time
import numpy as np
import torch
from typing import List, Tuple, Optional

import config
from source.tokenizer.tokenizer2025 import Tokenizer2025
from source.inference.beamsearch import BeamSearchOptim
from source.build_model.model import Transformer2025
from source.train_model.util import load_checkpoint_onlymodel

CHECKPOINT_PATH = r"D:\chuyen_nganh\Machine Translation version2\Saved\checkpoint_799999_epoch_1.pt"
BEAM_WIDTH = 5
TEST_SEQUENCES = [
    "Upon returning to her home in Toronto, Ontario, she began training to become a bodybuilder",
    "After approximately 10 minutes , Murray stated he left Jackson 's side to go to the restroom",
    "We have professional technician for loading Guaranteed the goods load into container without any damage",
    "Lam Dong is a beautiful town, captivates all those who have been there once"
] * 4

class MTInference:
    def __init__(self, 
                 checkpoint_path: str,
                 beam_width: int = 5,
                 max_len: int = None,
                 device: str = None):
        
        self.device = device or config.DEVICES
        self.max_len = max_len or config.MAX_LEN_INFERENCE
        self.checkpoint_path = checkpoint_path
        self.beam_width = beam_width
        
        self.tokenizer: Optional[Tokenizer2025] = None
        self.model: Optional[Transformer2025] = None
        self.beam_search: Optional[BeamSearchOptim] = None
        
        print(f"Initializing Model on {self.device}...")
        self._initialize_components()
        print("Model loaded successfully.")
    
    def _initialize_components(self):
        self.tokenizer = Tokenizer2025(
            model_spm_path=config.MODEL_SPM_PATH, 
            legacy=False
        )
        
        self.model = Transformer2025().to(device=self.device)
        load_checkpoint_onlymodel(self.checkpoint_path, model=self.model)
        self.model.eval() 
        
        self.beam_search = BeamSearchOptim(
            beam_width=self.beam_width,
            max_len=self.max_len,
            sos_id=config.BOS_TOKEN,
            eos_id=config.EOS_TOKEN,
            device=self.device
        )
    
    def _prepare_batch(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(sequences)
        sequences_encoded, _ = self.tokenizer.encode(sequences)
        max_length_src = max(map(len, sequences_encoded))
        padding_token_id = self.tokenizer.get_pad_token()[0]
        
        encoded_padded = np.full(
            (batch_size, max_length_src),
            fill_value=padding_token_id,
            dtype=np.int64
        )
        
        for i, arr in enumerate(sequences_encoded):
            encoded_padded[i, :len(arr)] = arr
        
        key_padding_mask = (encoded_padded != padding_token_id)
        
        encoded_tensor = torch.from_numpy(encoded_padded).to(self.device)
        mask_tensor = torch.from_numpy(key_padding_mask).to(self.device)
        
        return encoded_tensor, mask_tensor
    
    def translate(self, sequences: List[str], use_cache: bool = True) -> Tuple[List[str], List[float]]:
        if not sequences:
            raise ValueError("Danh sách câu đầu vào không được để trống.")
        sequences = [text + "." for text in sequences]
        encoded_padded, key_padding_mask = self._prepare_batch(sequences)
        
        with torch.inference_mode():
            sequences_result, scores = self.beam_search.batch_translate(
                encoded_padded,
                model=self.model,
                source_mask=key_padding_mask,
                use_cache=use_cache
            )
            
        decoded_results = self.tokenizer.decode(sequences_result, skip_special_tokens=True)
        return decoded_results, scores

    def cleanup(self):
        if self.model is not None:
            del self.model
        if self.beam_search is not None:
            del self.beam_search
        torch.cuda.empty_cache()

def run_demo(translator: MTInference, sequences: List[str]):
    print("\n" + "="*20 + " DEMO TRANSLATION " + "="*20)
    start = time.time()
    translated, scores = translator.translate(sequences, use_cache=True)
    end = time.time()
    
    for src, tgt, score in zip(sequences, translated, scores):
        print(f"Src: {src}")
        print(f"Tgt: {tgt}")
        print(f"Score: {score:.4f}")
        print("-" * 10)
    print(f"Demo time: {end - start:.4f}s")

def run_benchmark(translator: MTInference, sequences: List[str], n_runs: int = 10):
    print(f"\n" + "="*20 + f" BENCHMARK (Runs: {n_runs}, Batch: {len(sequences)}) " + "="*20)
    print("Heating up GPU...")
    translator.translate(sequences, use_cache=True)
    torch.cuda.synchronize()
    
    total_time_no_cache = 0
    for _ in range(n_runs):
        start = time.time()
        translator.translate(sequences, use_cache=False)
        torch.cuda.synchronize()
        end = time.time()
        total_time_no_cache += (end - start)
    
    avg_no_cache = total_time_no_cache / n_runs
    print(f"Without Cache (Avg): {avg_no_cache:.4f}s / batch")

    total_time_cache = 0
    for _ in range(n_runs):
        start = time.time()
        translator.translate(sequences, use_cache=True)
        torch.cuda.synchronize()
        end = time.time()
        total_time_cache += (end - start)
    
    avg_cache = total_time_cache / n_runs
    print(f"With Cache    (Avg): {avg_cache:.4f}s / batch")
    
    if avg_cache < avg_no_cache:
        speedup = (avg_no_cache - avg_cache) / avg_no_cache * 100
        print(f"Speedup: {speedup:.2f}% <=> avg_cache/avg_without_cache: {avg_no_cache/avg_cache:.2f}x")
    else:
        print("Cache is slower (Check implementation overhead)")

if __name__ == "__main__":
    try:
        translator = MTInference(
            checkpoint_path=CHECKPOINT_PATH,
            beam_width=BEAM_WIDTH
        )
        
        run_demo(translator, TEST_SEQUENCES[:4])
        
        # run_benchmark(translator, TEST_SEQUENCES, n_runs=10)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'translator' in locals():
            translator.cleanup()