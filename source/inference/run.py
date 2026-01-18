"""
NƠI CHẠY MODEL - THỰC NGHIỆM KẾT QUẢ
Machine Translation Inference Module with Class-based Structure
"""
import numpy as np
import torch
from source.tokenizer.tokenizer2025 import Tokenizer2025
from source.inference.beamsearch import BeamSearchOptim
from source.build_model.model import Transformer2025
from source.train_model.util import load_checkpoint_onlymodel
import config

class MTInference:
    def __init__(self, 
                 checkpoint_path: str = r"D:\chuyen_nganh\Machine Translation version2\Saved\checkpoint1.pt",
                 beam_width: int = 5,
                 max_len: int = None,
                 device: str = None):
        self.device = device or config.DEVICES
        self.max_len = max_len or config.MAX_LEN_INFERENCE
        self.checkpoint_path = checkpoint_path
        self.beam_width = beam_width
        
        self.tokenizer = None
        self.model = None
        self.beam_search = None
        
        self._initialize_components()
    
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
    
    def _prepare_batch(self, sequences: list) -> tuple:
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
        
        encoded_padded = torch.from_numpy(encoded_padded).to(self.device)
        key_padding_mask = torch.from_numpy(key_padding_mask).to(self.device)
        
        return encoded_padded, key_padding_mask
    
    def translate(self, sequences: list) -> tuple:
        if not sequences:
            raise ValueError("sequences không được để trống")
        
        encoded_padded, key_padding_mask = self._prepare_batch(sequences)
        
        with torch.no_grad():
            sequences_result, scores = self.beam_search.batch_translate(
                encoded_padded,
                model=self.model,
                source_mask=key_padding_mask,
                use_cache=True
            )
        
        decoded_results = self.tokenizer.decode(sequences_result, skip_special_tokens=False)
        return decoded_results, scores
    
    def translate_single(self, sequence: str) -> str:
        results, _ = self.translate([sequence])
        return results[0]
    
    def __del__(self):
        if self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache()

def demo_inference():
    sequences = [
        "sequence A",
        "sequence B",
    ]
    translator = MTInference(beam_width=5)
    translated, scores = translator.translate(sequences)
    
    print("Input sequences:")
    for seq in sequences:
        print(f"  - {seq}")
    
    print("\nTranslated sequences:")
    for trans in translated:
        print(f"  - {trans}")
    
    print("\nScores:")
    for score in scores:
        print(f"  - {score}")
    
    single_result = translator.translate_single("Hello world")
    print(f"\nSingle translation: {single_result}")

if __name__ == "__main__":
    demo_inference()