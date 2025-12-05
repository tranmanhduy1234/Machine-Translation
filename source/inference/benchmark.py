import torch
import time
from transformers import MarianMTModel, MarianTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def _sync_if_cuda():
    if device.type == "cuda":
        torch.cuda.synchronize()

# ======== LOAD MODELS ========
model_name = "Helsinki-NLP/Opus-MT-vi-en"  # Vietnamese to English
print(f"Model: {model_name}")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model_marian = MarianMTModel.from_pretrained(model_name).to(device).eval()

# ======== INPUT ========
text = 'Sự chuyển dịch mô hình (paradigm shift) trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP) hiện đại không chỉ đơn thuần là sự thay đổi về thuật toán tối ưu hóa, mà là một cuộc cách mạng về tư duy biểu diễn tri thức. Trước năm 2017, các kiến trúc mạng nơ-ron tái phát (Recurrent Neural Networks - RNN) và biến thể bộ nhớ dài-ngắn hạn (LSTM) thống trị việc xử lý dữ liệu dạng chuỗi. Tuy nhiên, bản chất xử lý tuần tự (sequential processing) của chúng tạo ra một "nút thắt cổ chai" nghiêm trọng về khả năng tính toán song song và giới hạn trong việc nắm bắt các phụ thuộc xa (long-term dependencies). Vấn đề "vanishing gradient" (biến mất đạo hàm), mặc dù đã được giảm thiểu bởi cơ chế cổng (gating) của LSTM, vẫn là một rào cản lớn khi ngữ cảnh đầu vào mở rộng quá giới hạn bộ nhớ làm việc.Bước ngoặt lịch sử xuất hiện với sự ra đời của kiến trúc Transformer, được giới thiệu trong bài báo cáo khoa học "Attention Is All You Need". Thay vì xử lý thông tin theo trục thời gian tuyến tính từ trái sang phải, Transformer loại bỏ hoàn toàn sự tái phát để thay thế bằng cơ chế Self-Attention (Tự chú ý). Về mặt toán học, cơ chế này cho phép mô hình tính toán trọng số quan hệ giữa mọi cặp token trong câu cùng một lúc, bất kể khoảng cách vị trí giữa chúng. Công thức cốt lõi:$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$không chỉ là một phép tính ma trận thuần túy, mà là sự mô phỏng cách con người tư duy ngữ nghĩa: khi đọc từ "ngân hàng", ta cần ánh xạ ngay lập tức tới ngữ cảnh của từ "sông" hay "tài chính" ở bất kỳ đâu trong đoạn văn để định nghĩa nó, thay vì phải chờ đợi tín hiệu lan truyền qua từng bước thời gian.Sự ưu việt của Transformer nằm ở khả năng song song hóa tối đa (parallelization). Điều này cho phép các nhà nghiên cứu tận dụng sức mạnh tính toán của phần cứng GPU hiện đại để huấn luyện trên những tập dữ liệu khổng lồ (hàng nghìn tỷ token), dẫn đến sự ra đời của các Mô hình Ngôn ngữ Lớn (LLM) như BERT, GPT hay Llama. Khả năng "scaling laws" (định luật quy mô) đã được chứng minh thực nghiệm: khi tăng dữ liệu và tham số, hiệu suất mô hình tăng theo quy luật lũy thừa, mở ra khả năng suy luận (reasoning) và khái quát hóa (generalization/zero-shot learning) mà các mô hình thống kê trước đây không thể chạm tới.Tóm lại, sự chuyển đổi từ RNN sang Transformer đại diện cho việc chuyển từ tư duy tuyến tính, cục bộ sang tư duy phi tuyến tính, toàn cục và đa chiều. Nó đặt nền móng vững chắc cho Trí tuệ Nhân tạo Tổng quát (AGI) bằng cách cung cấp một kiến trúc đủ linh hoạt để học các phân phối xác suất phức tạp của ngôn ngữ và tri thức nhân loại, biến đổi cách máy móc "hiểu" thế giới từ việc khớp mẫu đơn giản sang việc nắm bắt các cấu trúc ngữ nghĩa sâu sắc (deep semantic structures).'

print(f"\nDộ lớn đầu vào: {len(text)} ký tự")
print(f"Văn bản gốc (Tiếng Việt):\n{text}\n")

# ======== WARM-UP ========
print("Warming up...")
for _ in range(3):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        model_marian.generate(inputs["input_ids"], max_length=512)

# ======== BENCHMARK TRANSLATION ========
def benchmark_translation(name, text):
    _sync_if_cuda()
    start = time.time()
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        output_ids = model_marian.generate(inputs["input_ids"], max_length=512)
        translated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    _sync_if_cuda()
    end = time.time()
    
    elapsed_ms = (end - start) * 1000
    print(f"\n{name}:")
    print(f"Thời gian: {elapsed_ms:.2f} ms")
    print(f"Văn bản dịch (Tiếng Anh):\n{translated_text}")
    
    return elapsed_ms

# ======== PARAM COUNT ========
params_marian = sum(p.numel() for p in model_marian.parameters())
print(f"\nMarianMT params: {params_marian:,}")

print("\n===== BENCHMARK =====")
benchmark_translation("Translation", text)