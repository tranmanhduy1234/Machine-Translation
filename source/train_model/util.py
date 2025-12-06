def logGradient_histogram_mean_std(model, writer): # Log gradient truyền qua từng lớp. - dạng histogram
    pass
def logWeight_histogram_mean_std(model, writer): # Bản đồ weight. - dạng histogram
    pass
def logBias_histogram_mean_std(model, writer):
    pass
def logLoss(): # Loss qua các steps. - add_scaler
    pass
def logBLEU(): # sử dụng torchmetrics đánh giá BLEU.
    pass
def singleValueSpectrum(): # bản đồ xương sống.
    pass
def logLearningRate(): # add_scaler
    pass
def logEmbedding():
    pass
def logText():
    """
        **Step {step} Summary:**
        - Loss: {loss:.4f}
        - Accuracy: {accuracy:.2f}%
        - CPU Usage: {cpu_usage:.1f}%
        - RAM Usage: {ram_usage:.1f}%
        - Gradient Mean: {dummy_grads.mean().item():.6f}
    """
    pass
def logPerformance(): # Đo hiệu suất hệ thống: CPU, GPU, RAM, I/O
    pass

"""
    Gradient noise scale (GNS)
    Hessian spectrum
    NTK monitoring
    Token-level BLEU per layer
    Attention entropy visualization
"""