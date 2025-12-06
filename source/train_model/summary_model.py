import os

import torch
import time
from source.build_model.model import Transformer2025
from torch.utils.tensorboard import SummaryWriter
import datetime
writer = SummaryWriter(f'source/train_model/summary/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

model = Transformer2025()
model.eval()
print(f"Tổng lượng tham số mô hình: {model.count_parameters()}")
exit(0)
with torch.no_grad():
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, global_step=0)
        
print("Đã ghi log khởi tạo xong. Hãy kiểm tra TensorBoard.")
writer.close()