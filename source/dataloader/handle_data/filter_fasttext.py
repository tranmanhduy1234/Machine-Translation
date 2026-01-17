import fasttext
import csv
import os
from urllib.request import urlretrieve

# Tải model fasttext nếu chưa có
MODEL_PATH = 'lid.176.bin'
if not os.path.exists(MODEL_PATH):
    print("Đang tải model fastText...")
    urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', MODEL_PATH)
    print("Đã tải xong model!")

# Load model
print("Đang load model fastText...")
model = fasttext.load_model(MODEL_PATH)
print("Model đã sẵn sàng!")

def detect_language(text, threshold=0.5):
    """
    Phát hiện ngôn ngữ của text
    Returns: (language_code, confidence)
    """
    text = text.replace('\n', ' ').strip()
    if not text:
        return None, 0.0
    
    predictions = model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    conf = predictions[1][0]
    
    return lang, conf

def filter_envi_pairs(input_file, output_file, threshold=0.5):
    """
    Lọc các cặp câu en-vi từ file TSV
    - input_file: đường dẫn file TSV đầu vào
    - output_file: đường dẫn file TSV đầu ra (các cặp hợp lệ)
    - threshold: ngưỡng confidence tối thiểu (0-1)
    """
    valid_pairs = []
    invalid_pairs = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Đọc header
        
        for idx, row in enumerate(reader, 1):
            if len(row) < 2:
                print(f"Dòng {idx}: Không đủ 2 cột, bỏ qua")
                invalid_pairs.append((idx, row, "Thiếu cột"))
                continue
            
            en_text = row[0].strip()
            vi_text = row[1].strip()
            
            # Detect ngôn ngữ
            en_lang, en_conf = detect_language(en_text, threshold)
            vi_lang, vi_conf = detect_language(vi_text, threshold)
            
            # Kiểm tra điều kiện
            is_valid = (en_lang == 'en' and en_conf >= threshold and 
                       vi_lang == 'vi' and vi_conf >= threshold)
            
            if is_valid:
                valid_pairs.append(row)
            else:
                reason = f"EN: {en_lang}({en_conf:.2f}), VI: {vi_lang}({vi_conf:.2f})"
                invalid_pairs.append((idx, row, reason))
                
            if idx % 1000 == 0:
                print(f"Đã xử lý {idx} dòng...")
    
    # Ghi file output
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(valid_pairs)
    
    # In thống kê
    print("\n" + "="*50)
    print(f"Tổng số cặp: {len(valid_pairs) + len(invalid_pairs)}")
    print(f"Cặp hợp lệ: {len(valid_pairs)}")
    print(f"Cặp không hợp lệ: {len(invalid_pairs)}")
    print(f"Tỷ lệ hợp lệ: {len(valid_pairs)/(len(valid_pairs)+len(invalid_pairs))*100:.2f}%")
    
    # Hiển thị một vài ví dụ không hợp lệ
    if invalid_pairs:
        print("\nMột số ví dụ không hợp lệ:")
        for idx, row, reason in invalid_pairs[:5]:
            print(f"\nDòng {idx}: {reason}")
            print(f"  EN: {row[0][:100]}")
            print(f"  VI: {row[1][:100]}")
    
    print(f"\nĐã lưu kết quả vào: {output_file}")

# Sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn
    input_file = "D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetEVBCorpus.tsv"  # File TSV đầu vào
    output_file = "D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetEVBCorpus_rs.tsv"  # File TSV đầu ra
    
    # Lọc với ngưỡng confidence 0.5
    filter_envi_pairs(input_file, output_file, threshold=0.5)