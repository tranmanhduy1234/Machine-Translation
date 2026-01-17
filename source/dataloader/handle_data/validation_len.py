import matplotlib.pyplot as plt
from collections import Counter
import sentencepiece as spm
from tqdm import tqdm

# Tính độ dài câu theo batch
def calculate_lengths_from_file(filename, sp_model, batch_size=100000):
    """
    Đọc file và tính độ dài theo batch để tiết kiệm memory
    Trả về danh sách độ dài cho câu nguồn và câu đích
    """
    source_lengths = []
    target_lengths = []
    
    print(f"Đang xử lý file: {filename}")
    
    # Đếm tổng số dòng để hiển thị progress
    print("Đang đếm số dòng...")
    with open(filename, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    # Xử lý từng batch
    with open(filename, 'r', encoding='utf-8') as f:
        source_batch = []
        target_batch = []
        
        for line in tqdm(f, total=total_lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) == 2:
                source_batch.append(parts[0])
                target_batch.append(parts[1])
                
                # Khi đủ batch_size thì tokenize
                if len(source_batch) >= batch_size:
                    # Tokenize batch
                    for sent in source_batch:
                        ids = sp_model.EncodeAsIds(sent)
                        source_lengths.append(len(ids))
                    
                    for sent in target_batch:
                        ids = sp_model.EncodeAsIds(sent)
                        target_lengths.append(len(ids))
                    
                    # Clear batch
                    source_batch = []
                    target_batch = []
        
        # Xử lý batch cuối cùng
        if source_batch:
            for sent in source_batch:
                ids = sp_model.EncodeAsIds(sent)
                source_lengths.append(len(ids))
            
            for sent in target_batch:
                ids = sp_model.EncodeAsIds(sent)
                target_lengths.append(len(ids))
    
    return source_lengths, target_lengths

# Thống kê phân bố
def get_statistics(lengths):
    """Thống kê các chỉ số về độ dài"""
    if not lengths:
        return {}
    
    sorted_lengths = sorted(lengths)
    return {
        'min': min(lengths),
        'max': max(lengths),
        'mean': sum(lengths) / len(lengths),
        'median': sorted_lengths[len(lengths) // 2],
        'percentile_90': sorted_lengths[int(len(lengths) * 0.9)],
        'percentile_95': sorted_lengths[int(len(lengths) * 0.95)],
        'percentile_99': sorted_lengths[int(len(lengths) * 0.99)],
        'total_sentences': len(lengths)
    }

# Vẽ biểu đồ phân bố
def plot_distribution(source_lengths, target_lengths, max_length=None):
    """Vẽ biểu đồ phân bố độ dài của cả hai phần"""
    
    # Giới hạn độ dài để vẽ (loại bỏ outliers)
    if max_length is None:
        max_length = max(
            int(sorted(source_lengths)[int(len(source_lengths) * 0.99)]),
            int(sorted(target_lengths)[int(len(target_lengths) * 0.99)])
        )
    
    source_filtered = [l for l in source_lengths if l <= max_length]
    target_filtered = [l for l in target_lengths if l <= max_length]
    
    print(f"\nVẽ biểu đồ với max_length={max_length}")
    print(f"Số câu nguồn được vẽ: {len(source_filtered)}/{len(source_lengths)}")
    print(f"Số câu đích được vẽ: {len(target_filtered)}/{len(target_lengths)}")
    
    plt.figure(figsize=(14, 6))
    
    # Tính phân bố
    source_counter = Counter(source_filtered)
    target_counter = Counter(target_filtered)
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist([source_filtered, target_filtered], bins=50, alpha=0.7, 
             label=['Câu nguồn', 'Câu đích'], color=['blue', 'orange'])
    plt.xlabel('Độ dài (số token)')
    plt.ylabel('Số lượng câu')
    plt.title(f'Phân bố độ dài câu - Histogram (≤{max_length} tokens)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Line plot
    plt.subplot(1, 2, 2)
    
    # Chuẩn bị dữ liệu cho line plot
    source_x = sorted(source_counter.keys())
    source_y = [source_counter[x] for x in source_x]
    target_x = sorted(target_counter.keys())
    target_y = [target_counter[x] for x in target_x]
    
    plt.plot(source_x, source_y, marker='o', markersize=2, 
             label='Câu nguồn', color='blue', alpha=0.7, linewidth=1)
    plt.plot(target_x, target_y, marker='s', markersize=2, 
             label='Câu đích', color='orange', alpha=0.7, linewidth=1)
    plt.xlabel('Độ dài (số token)')
    plt.ylabel('Số lượng câu')
    plt.title(f'Phân bố độ dài câu - Line Plot (≤{max_length} tokens)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Hàm chính
def main(data_filename, sp_model_path, batch_size=10000, max_plot_length=None):
    """Hàm chính để chạy phân tích"""
    print(f"Đang load SentencePiece model: {sp_model_path}")
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    print(f"Vocab size: {sp.GetPieceSize()}")
    
    # Tính độ dài theo token với batch processing
    source_lengths, target_lengths = calculate_lengths_from_file(
        data_filename, sp, batch_size=batch_size
    )
    
    print(f"\nĐã xử lý {len(source_lengths)} cặp câu")
    
    # Thống kê
    print("\n" + "="*60)
    print("THỐNG KÊ CÂU NGUỒN:")
    print("="*60)
    source_stats = get_statistics(source_lengths)
    for key, value in source_stats.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    print("\n" + "="*60)
    print("THỐNG KÊ CÂU ĐÍCH:")
    print("="*60)
    target_stats = get_statistics(target_lengths)
    for key, value in target_stats.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    # Vẽ biểu đồ
    print("\n" + "="*60)
    print("Đang vẽ biểu đồ phân bố...")
    print("="*60)
    plot_distribution(source_lengths, target_lengths, max_length=max_plot_length)
    print("\nĐã lưu biểu đồ vào file: length_distribution.png")

# Chạy chương trình
if __name__ == "__main__":
    # Cấu hình
    data_filename = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetEVBCorpus.tsv"        # File tsv dữ liệu song ngữ
    sp_model_path = r"D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model"     # File SentencePiece model
    batch_size = 1000000                     # Số câu xử lý mỗi batch (tăng nếu RAM đủ)
    max_plot_length = 1000                 # None = tự động (99 percentile), hoặc set số cụ thể
    main(data_filename, sp_model_path, batch_size, max_plot_length)