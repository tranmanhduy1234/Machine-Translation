from source.build_model.model import Transformer2025
import torch
import torch.nn as nn
from typing import Tuple, Optional

class BeamSearchOptimizer(nn.Module):
    def __init__(self, beam_width, max_len, sos_id, eos_id, device="cuda", alpha=0.6, per_beam_k=None):
        super().__init__()
        self.beam_width = beam_width
        self.max_len = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        self.alpha = alpha
        self.per_beam_k = per_beam_k
    
    @torch.no_grad()
    def translate(self, inputs_id, model: Transformer2025, source_mask=None):
        batch_size = inputs_id.shape[0]
        
        with torch.cuda.nvtx.range("encoder"):
            encoder_output = model.inference_embed_encoder(inputs_id=inputs_id, src_kpmask=source_mask)
            encoder_output = encoder_output.expand(self.beam_width, -1, -1).contiguous()
            source_mask = source_mask.expand(self.beam_width, -1).contiguous()
            
        beam_seqs = torch.full((self.beam_width, 1), self.sos_id, dtype=torch.long, device=self.device)
        beam_scores = torch.zeros(self.beam_width, device=self.device)
        finished = torch.zeros(self.beam_width, dtype=torch.bool, device=self.device)
        beam_lengths = torch.ones(self.beam_width, dtype=torch.long, device=self.device)
        
        past_kv = None
        
        for step in range(self.max_len):
            tokens_to_embed = beam_seqs if step == 0 else beam_seqs[:, -1:]
            
            with torch.cuda.nvtx.range("embed_decode"):
                token_embed = model.inference_embedding_layer(tokens_to_embed)
                
                logits, past_kv = model.inference_decoder_projection_with_cache(
                    input_decoder=token_embed,
                    encoder_output=encoder_output,
                    tgt_kpmask=None,
                    src_kpmask=source_mask,
                    past_kv=past_kv,
                    use_cache=True
                )
            
            next_token_logits = logits[:, -1, :] # [beam_width, vocab_size]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            if finished.any():
                log_probs = log_probs.clone()
                log_probs[finished, :] = torch.finfo(log_probs.dtype).min
                log_probs[finished, self.eos_id] = 0.0
            
            vocab_size = log_probs.shape[-1]
            k = self.per_beam_k or min(vocab_size, self.beam_width * 4)
            
            topk_vals, topk_ids = torch.topk(log_probs, k, dim=-1) # beamwidth, k
            
            cand_scores = beam_scores.unsqueeze(1) + topk_vals # beam width, kkk
            flat_scores = cand_scores.view(-1)
            
            topk_flat_scores, topk_flat_indices = torch.topk(flat_scores, self.beam_width)
            
            parent_beam_indices = topk_flat_indices // k
            chosen_token_positions = topk_flat_indices % k
            chosen_token_indices = topk_ids[parent_beam_indices, chosen_token_positions]
            
            # update sequenses
            beam_scores = topk_flat_scores
            beam_seqs = torch.cat([beam_seqs[parent_beam_indices], chosen_token_indices.unsqueeze(1)], dim=-1)
            
            # update finished status ans length
            is_eos = (chosen_token_indices == self.eos_id)
            finished = finished[parent_beam_indices] | is_eos
            beam_lengths = beam_lengths[parent_beam_indices].clone()
            beam_lengths[~finished] += 1
            
            if past_kv is not None:
                past_kv = self._reorder_cache(past_kv, parent_beam_indices)
            
            if finished.all():
                break
        
        # length penalty
        length_penalty = torch.pow((5.0 + beam_lengths.float()) / 6.0, self.alpha)
        final_scores = beam_scores / length_penalty
        best_idx = torch.argmax(final_scores)
        return beam_seqs[best_idx], final_scores[best_idx]

    @staticmethod
    def _reorder_cache(past_kv, beam_indices):
        if past_kv is None:
            return None
    
        reordered_past = []
        for layer_past in past_kv:
            # layer_past: (k_cache, v_cache) each [beamwidth, num_head, sseq_len, head_dim]
            k_cache, v_cache = layer_past
            reordered_past.append((
                k_cache[beam_indices],
                v_cache[beam_indices]
            ))
        return reordered_past
    
    @staticmethod
    def _reorder_cache_batch(past_kv, global_indices, batch_size):
        if past_kv is None:
            return None
        
        reordered_past = []
        for layer_past in past_kv:
            k_cache, v_cache = layer_past
            reordered_past.append((
                k_cache[global_indices],
                v_cache[global_indices]
            ))
        return reordered_past
    
if __name__=="__main__":
    import sentencepiece as spm
    import time
    
    sp = spm.SentencePieceProcessor()
    sp.Load(r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model')
    
    input_ids = torch.randint(0, 40000, (1, 256)).to("cuda")
    batch_ids = torch.randint(0, 40000, (16, 256)).to("cuda")
    source_mask = ~torch.zeros((1, 256), dtype=bool, device="cuda")
    
    model = Transformer2025().to("cuda")
    model.eval()
    with torch.no_grad():
        beamsearchhead = BeamSearchOptimizer(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        _, _ = beamsearchhead.translate(input_ids, model, source_mask)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            seq, score = beamsearchhead.translate(inputs_id=input_ids, model=model, source_mask=source_mask)
            
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(elapsed)