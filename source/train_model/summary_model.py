import os

import torch
import time
from source.build_model.model import Transformer2025
from torch.utils.tensorboard import SummaryWriter
import datetime
# 2. Khởi tạo Writer
writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

model = Transformer2025()
model.eval()

# 3. Dùng with torch.no_grad() để tiết kiệm bộ nhớ khi đọc tham số
with torch.no_grad():
    for i in range(100):
        for name, param in model.named_parameters():
            # Thêm global_step=0 để đánh dấu đây là trạng thái khởi tạo
            writer.add_histogram(f'Weights/{name}', param, global_step=i)
        
print("Đã ghi log khởi tạo xong. Hãy kiểm tra TensorBoard.")
writer.close() # Luôn nhớ đóng writer để flush dữ liệu xuống ổ cứng