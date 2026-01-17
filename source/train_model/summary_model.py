"""
    CHẠY CÁC BẢN ĐÁNH GIÁ MODEL VỀ THAM SỐ, TINH CHỈNH MÔ HÌNH.
"""
from source.build_model.model import Transformer2025
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv
from source.train_model.util import *
writer = SummaryWriter(f'source/train_model/summary/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
model = Transformer2025()
def export_params_to_csv(model, csv_path=r"D:\chuyen_nganh\Machine Translation version2\source\architecture\aversion1.csv"):
    rows = []
    rows.append(["layer_name", "layer_type", "num_params"])
    for name, module in model.named_modules():
            # Lấy danh sách ID của các tham số để kiểm tra trùng
            param_ids = [str(id(p)) for p in module.parameters(recurse=False) if p.requires_grad]
            params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if params > 0:
                rows.append([
                    name,
                    module.__class__.__name__,
                    params,
                    " | ".join(param_ids) # <--- Thêm cột này
                ])
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"✅ Saved to {csv_path}")
# logWeightBias_histogram_mean_std(model=model, writer=writer, index=1)
# print(model.count_parameters())

# warmup
model = model.to("cuda")
model.eval()
model = model
with torch.no_grad():
    inputids = torch.rand(16, 256, 640).to("cuda")
    for i in range(10):
        model.inference_decoder_layer(inputids, inputids,None, None,
                                      True, False, True)
torch.cuda.synchronize()
import time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# start1 = time.time()
# with torch.no_grad():
#     for i in range(512):
#         ids = torch.rand(16, i + 1, 640).to("cuda").half()
#         model.inference_decoder_layer(ids, ids, None, None, True, False, False)
# end1 = time.time() - start1

model.reset_cache()
inputids = torch.randint(0, 4000, (16, 256)).to("cuda")
decoderout = torch.rand(16, 256, 640).to("cuda")
with torch.no_grad():
    start.record()
    model.inference_embed_encoder(inputids, None, is_causal=False)
    for i in range(256):
        ids = torch.rand(16, i + 1, 640).to("cuda")
        model.inference_decoder_layer(ids, ids, None, None, True, False, False)
    model.inference_output_projection(decoderout)
    end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)

print(f"Time: {elapsed_ms:.2f} ms")
print(f"Batch size: 16")
print(f"Avg per sample: {elapsed_ms / 16:.2f} ms")