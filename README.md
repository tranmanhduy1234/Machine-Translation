# Mô hình dịch máy tự xây dựng Transformer2025
Xây dựng: Trần Mạnh Duy
Những kỹ thuật được sử dụng so với kiến trúc gốc trong bài báo
Attention is all you need
- Pre-norm được sử dụng thay cho post-norm truyền thống
- FlashAttention, gom ma trận tăng tốc độ xử lý
- Embedding learnable
- Khởi tạo trọng số ban đầu xavier
- Thay đổi hàm kích hoạt ở FFN sang switch (SILU)