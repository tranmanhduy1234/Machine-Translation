# Mô hình dịch máy tự xây dựng Transformer2025
## 📋 Mô tả dự án
✨ Kiến trúc gốc: Transformer 2017  
✨ Xây dựng: Trần Mạnh Duy
## 🚀Những kỹ thuật mới được sử dụng so với kiến trúc gốc trong bài báo Attention is all you need
- Pre-norm được sử dụng thay cho post-norm truyền thống
- FlashAttention, gom ma trận tăng tốc độ xử lý
- Embedding learnable
- Khởi tạo trọng số ban đầu xavier
- Thay đổi hàm kích hoạt ở FFN sang GELU - thay vì RELU

## 🔧Các vấn đề trong quá trình xây dựng
- Xây dựng kiến trúc Transformer ✅
- Xây dựng head predict sử dụng beemsearch ✅
- Feature Engineering & Preprocessing
- Data cleaning: remove NaN, handle outliers
- Normaiization/Standaridization
- Dataloader xử lý, gom dữ liệu trước khi đưa vào mô hình.
- Loss Function và Optimizer
- Training code Forward Pass / Backward Pass - Gradient Descent - Epoch / Batch / Iteration - Backpropagation - Regularization: L1, L2, Dropout - Early Stopping
- Validation & Hyperparameter Tuning
- Search Methods: Grid Search, Random Search, Bayesian Optimization
- Có thể trừng phạt trọng số riêng cho từng lớp
- Train
- Evaluation
- Monitoring & Inference
- Cân nhắc chuyển đổi LayerNorm qua RMSNorm