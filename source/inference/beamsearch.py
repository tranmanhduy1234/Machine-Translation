"""
    CẦN BẢN CẢI TIẾN CÓ SỬ DỤNG CACHE - FUSED TỐI ƯU TỐC ĐỘ INFERENCE
"""
from source.build_model.model import Transformer2025
import torch
import torch.nn as nn

# Về cơ bản là sử dụng được, tuy nhiên có thể tối ưu thêm
class BeamSearchOptim(nn.Module):
    def __init__(self, beam_width, max_len, sos_id, eos_id, device='cuda', alpha=0.6, per_beam_k=None):
        super().__init__()
        self.B = beam_width
        self.max_len = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        self.alpha = alpha
        self.per_beam_k = per_beam_k

    @torch.no_grad()
    def translate(self, inputs_id, model: Transformer2025, source_mask=None):
        # encoder once
        encoder_output = model.inference_embed_encoder(inputs_id=inputs_id, src_kpmask=source_mask)  # [1, seqLen, embed_dim]
        encoder_output = encoder_output.expand(self.B, -1, -1).contiguous()  # [beamwidth, seqLen, embed_dim]
        source_mask = source_mask.expand(self.B, -1).contiguous() # beamwidth, seq_len
        
        beam_seqs = torch.full((self.B, 1), self.sos_id, dtype=torch.long, device=self.device)  # [beam_width,1]
        beam_scores = torch.zeros(self.B, device=self.device)  # log probs: [beam_width]
        finished = torch.zeros(self.B, dtype=torch.bool, device=self.device) # [beam_width]
        beam_lengths = torch.ones(self.B, dtype=torch.long, device=self.device) # [beam_width]

        for _ in range(self.max_len):
            # get embeddings for current beams
            beam_seqs_embed = model.inference_embedding_layer(beam_seqs) # [beam_width, seq_len, embed_dim]
            logits = model.inference_decoder_projection(input_decoder=beam_seqs_embed, 
                                                             encoder_output=encoder_output, 
                                                             tgt_kpmask=None, 
                                                             src_kpmask=source_mask)  # [beam_width, seq, vocab_size]
            next_token_logits = logits[:, -1, :]  # [Beam width, Vocab_size]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)  # [beam width, vocab size]
            
            # prevent expansion of finished beams: set all tokens -inf except EOS set 0
            if finished.any():
                log_probs = log_probs.clone()
                log_probs[finished, :] = -float("inf")
                log_probs[finished, self.eos_id] = 0.0

            _, vocab_size = log_probs.size()
            k = self.per_beam_k or min(vocab_size, self.B * 4)
            
            # per-beam topk for speed
            topk_vals, topk_ids = torch.topk(log_probs, k, dim=-1)  # both [B, k]
            # topk_vals [beam_width, k]
            # compute candidate scores: [B, k]
            cand_scores = beam_scores.unsqueeze(1) + topk_vals
            
            # flatten candidates: [B*k]
            flat_scores = cand_scores.view(-1)
            topk_flat_scores, topk_flat_indices = torch.topk(flat_scores, self.B)

            # topk_flat_indices are indices into flat_scores [B*k]
            # flat_scores[i] = cand_scores[i//k, i%k], where i = beam_idx * k + token_pos
            parent_beam_indices = topk_flat_indices // k  # Which beam (0 to B-1)
            chosen_token_positions = topk_flat_indices % k  # Which token in that beam's topk (0 to k-1)
            # Get the actual token IDs: topk_ids[beam_idx, token_pos]
            chosen_token_indices = topk_ids[parent_beam_indices, chosen_token_positions]

            # update beams
            beam_scores = topk_flat_scores
            beam_seqs = torch.cat([beam_seqs[parent_beam_indices], chosen_token_indices.unsqueeze(1)], dim=-1)

            is_eos = (chosen_token_indices == self.eos_id)
            finished = finished[parent_beam_indices] | is_eos

            beam_lengths = beam_lengths[parent_beam_indices].clone()
            beam_lengths[~finished] += 1

            if finished.all():
                break

        # apply length penalty
        float_lengths = beam_lengths.float()
        length_penalty = torch.pow((5.0 + float_lengths) / 6.0, self.alpha)
        final_scores = beam_scores / length_penalty
        best_idx = torch.argmax(final_scores)
        return (beam_seqs[best_idx], final_scores[best_idx])
            
    def batch_translate(self, batch_inputs_id, model: Transformer2025, source_mask=None):
        batch_size, _ = batch_inputs_id.shape
        # Encoder once
        encoder_output = model.inference_embed_encoder(inputs_id=batch_inputs_id, src_kpmask=source_mask)
        # [batch_size, seq_len_src, embed_dim]
        encoder_output = encoder_output.unsqueeze(1).expand(-1, self.B, -1, -1).contiguous()
        # [batch_size, beam_width, seq_len_src, embed_dim]
        encoder_output = encoder_output.reshape(batch_size * self.B, -1, encoder_output.shape[-1])
        # [batch_size * beam_width, seq_len_src, embed_dim]
        
        source_mask = source_mask.unsqueeze(1).expand(-1, self.B, -1).contiguous()
        # [batch_size, beam_width, seq_len_src]
        source_mask = source_mask.reshape(batch_size * self.B, -1)
        # [batch_size * beam_width, seq_len_src]

        beam_seqs = torch.full((batch_size * self.B, 1), self.sos_id, dtype=torch.long, device=self.device)
        beam_scores = torch.zeros((batch_size, self.B), device=self.device)
        finished = torch.zeros((batch_size, self.B), dtype=torch.bool, device=self.device)
        beam_lengths = torch.ones((batch_size, self.B), dtype=torch.long, device=self.device)

        for step in range(self.max_len):
            # Get embeddings for current beams
            beam_seqs_embed = model.inference_embedding_layer(beam_seqs)
            # [batch_size * beam_width, seq_len_tgt, embed_dim]
            
            logits = model.inference_decoder_projection(
                input_decoder=beam_seqs_embed, 
                encoder_output=encoder_output, 
                tgt_kpmask=None, 
                src_kpmask=source_mask
            )
            # [batch_size * beam_width, seq_len_tgt, vocab_size]
            
            next_token_logits = logits[:, -1, :]
            # [batch_size * beam_width, vocab_size]
            
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            # [batch_size * beam_width, vocab_size]
            
            # Reshape để xử lý per-batch
            log_probs = log_probs.view(batch_size, self.B, -1)
            # [batch_size, beam_width, vocab_size]
            
            # Prevent expansion of finished beams
            if finished.any():
                log_probs = log_probs.clone()
                log_probs[finished, :] = -float("inf")
                log_probs[finished, self.eos_id] = 0.0

            vocab_size = log_probs.shape[-1]
            k = self.per_beam_k or min(vocab_size, self.B * 4)
            
            # Per-beam topk
            topk_vals, topk_ids = torch.topk(log_probs, k, dim=-1)
            # [batch_size, beam_width, k]
            
            # Compute candidate scores
            cand_scores = beam_scores.unsqueeze(2) + topk_vals
            # [batch_size, beam_width, k]
            
            # Flatten và lấy top B candidates cho mỗi batch
            flat_scores = cand_scores.view(batch_size, -1)
            # [batch_size, beam_width * k]
            
            topk_flat_scores, topk_flat_indices = torch.topk(flat_scores, self.B, dim=-1)
            # [batch_size, beam_width]
            
            # Tìm parent beam và token position
            parent_beam_indices = topk_flat_indices // k
            # [batch_size, beam_width]
            
            chosen_token_positions = topk_flat_indices % k
            # [batch_size, beam_width]
            
            # Lấy token IDs từ topk_ids
            batch_indices = torch.arange(batch_size, device=self.device).view(batch_size, 1).expand(batch_size, self.B)
            chosen_token_indices = topk_ids[batch_indices, parent_beam_indices, chosen_token_positions]
            # [batch_size, beam_width]
            
            # Update beam sequences
            offsets = torch.arange(batch_size, device=self.device).view(batch_size, 1) * self.B
            global_parent_indices = parent_beam_indices + offsets
            # [batch_size, beam_width]
            
            global_parent_indices_flat = global_parent_indices.view(-1)
            beam_seqs = torch.cat([beam_seqs[global_parent_indices_flat], chosen_token_indices.view(-1, 1)], dim=-1)
            # [batch_size * beam_width, seq_len_tgt + 1]
            
            # Update finished status
            is_eos = (chosen_token_indices == self.eos_id)
            finished = finished[batch_indices, parent_beam_indices] | is_eos
            beam_lengths = beam_lengths[batch_indices, parent_beam_indices].clone()
            beam_lengths[~finished] += 1
            
            # Update scores
            beam_scores = topk_flat_scores 
            # [batch_size * beam_width]
            
            if finished.all():
                break

        # Apply length penalty
        float_lengths = beam_lengths.float()
        length_penalty = torch.pow((5.0 + float_lengths) / 6.0, self.alpha)
        final_scores = beam_scores / length_penalty
        
        # Lấy best sequence cho mỗi batch
        best_indices = torch.argmax(final_scores, dim=-1)
        # [batch_size]
        
        offsets = torch.arange(batch_size, device=self.device) * self.B
        global_best_indices = best_indices + offsets
        
        return beam_seqs[global_best_indices], final_scores[torch.arange(batch_size, device=self.device), best_indices]
        
if __name__=="__main__":
    import sentencepiece as spm 
    sp = spm.SentencePieceProcessor()
    sp.Load(r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model')
    
    sentence = """
        Chủ tịch Hồ Chí Minh đã từng bộc bạch lý tưởng cao đẹp của mình khi đất nước rơi vào hoàn cảnh khốn khó rằng: 
        “Tôi chỉ có một sự ham muốn, ham muốn tột bậc, là làm sao cho nước ta được hoàn toàn độc lập, dân ta được hoàn toàn tự do, 
        đồng bào ai cũng có cơm ăn áo mặc, ai cũng được học hành.” Ngày hôm nay, khi đất nước không còn phải chịu đựng những “tiếng bom rơi đạn nổ”, 
        dang đôi cánh vươn mình bay cao trên bầu trời hội nhập, ai sẽ là những người cầm lái, chèo chống để đưa con thuyền Việt Nam
        vượt qua sóng gió và vươn tới những chân trời mới? Câu trả lời chính là bạn – thế hệ trẻ của hôm nay.
        Vậy trách nhiệm của thế hệ trẻ trong kỷ nguyên vươn mình là gì?
    """
    batch_ids = torch.randint(0, 40000, (16, 256)).to("cuda")
    source_mask = ~torch.zeros((16, 256), dtype=bool, device="cuda")
    model = Transformer2025().to('cuda')
    model.eval()
    with torch.no_grad():
        beamsearchhead = BeamSearchOptim(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        rs, _ = beamsearchhead.batch_translate(batch_inputs_id=batch_ids, model=model, source_mask=source_mask)
        print(rs.shape)
    ids = sp.EncodeAsIds(sentence)
    inputs_id = torch.tensor(ids).unsqueeze_(0).to("cuda")
    source_mask = torch.zeros((inputs_id.shape), dtype=bool, device="cuda")
    with torch.no_grad():
        import time
        start = time.time()
        beamsearchhead = BeamSearchOptim(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        seq, score = beamsearchhead.translate(inputs_id=inputs_id, model=model, source_mask=~source_mask)
        print(seq.shape)
        result = sp.DecodeIds(seq.tolist())
        print(result)
        print(f"Total time: {time.time() - start}")