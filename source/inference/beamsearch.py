"""
    CẦN BẢN CẢI TIẾN CÓ SỬ DỤNG CACHE - FUSED TỐI ƯU TỐC ĐỘ INFERENCE
"""
from source.build_model.model import Transformer2025
import torch
import torch.nn as nn

# Về cơ bản là sử dụng được, tuy nhiên có thể tối ưu thêm
class BeamSearchOptim(nn.Module):
    def __init__(self, beam_width, model: Transformer2025, max_len, sos_id, eos_id, device='cuda', alpha=0.6, per_beam_k=None):
        super().__init__()
        self.B = beam_width
        self.model = model
        self.max_len = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        self.alpha = alpha
        self.per_beam_k = per_beam_k

    @torch.no_grad()
    def translate(self, inputs_id):
        # encoder once
        encoder_output = self.model.inference_embed_encoder(inputs_id=inputs_id, src_kpmask=None)  # [1, seqLen, embed_dim]
        encoder_output = encoder_output.expand(self.B, -1, -1).contiguous()  # [beamwidth, seqLen, embed_dim]

        beam_seqs = torch.full((self.B, 1), self.sos_id, dtype=torch.long, device=self.device)  # [beam_width,1]
        beam_scores = torch.zeros(self.B, device=self.device)  # log probs: [beam_width]
        finished = torch.zeros(self.B, dtype=torch.bool, device=self.device) # [beam_width]
        beam_lengths = torch.ones(self.B, dtype=torch.long, device=self.device) # [beam_width]

        for _ in range(self.max_len):
            # get embeddings for current beams
            beam_seqs_embed = self.model.inference_embedding_layer(beam_seqs) # [beam_width, seq_len, embed_dim]
            logits = self.model.inference_decoder_projection(input_decoder=beam_seqs_embed, encoder_output=encoder_output, tgt_kpmask=None, src_kpmask=None)  # [beam_width, seq, vocab_size]
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
    
    model = Transformer2025().to('cuda')
    model.eval()
    inputs_id = torch.tensor(sp.EncodeAsIds(sentence)).unsqueeze_(0).to("cuda")
    print(inputs_id.shape)
    with torch.no_grad():
        import time
        start = time.time()
        beamsearchhead = BeamSearchOptim(beam_width=5, model=model, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        seq, score = beamsearchhead.translate(inputs_id=inputs_id)
        print(seq.shape)
        result = sp.DecodeIds(seq.tolist())
        print(result)
        print(f"Total time: {time.time() - start}")