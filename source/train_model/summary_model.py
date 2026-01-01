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
# export_params_to_csv(model)
logWeightBias_histogram_mean_std(model=model, writer=writer, index=1)
print(model.count_parameters())