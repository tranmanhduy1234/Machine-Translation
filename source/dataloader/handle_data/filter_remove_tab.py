import csv
import sys

# Tăng giới hạn bộ nhớ cho các trường quá dài (thường gặp trong NLP)
csv.field_size_limit(sys.maxsize)

def clean_tsv(input_path, output_path, expected_columns=2):
    """
    input_path: Đường dẫn file gốc
    output_path: Đường dẫn file sau khi lọc
    expected_columns: Số cột chuẩn (ví dụ: 2 cho cặp câu src-tgt)
    """
    success_count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8', newline='') as f_in, \
         open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        
        # Dùng delimiter là tab
        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')
        
        print(f"Đang xử lý: {input_path}...")
        
        for i, row in enumerate(reader):
            # KIỂM TRA: Nếu số cột khớp với chuẩn thì giữ lại
            if len(row) == expected_columns:
                writer.writerow(row)
                success_count += 1
            else:
                # Nếu sai số cột (thừa hoặc thiếu tab) -> Bỏ qua
                error_count += 1
                # In thử 5 dòng lỗi đầu tiên để kiểm tra
                if error_count <= 5:
                    print(f"[Cảnh báo] Dòng {i+1} bị lỗi (có {len(row)} cột): {row}")

    print("="*30)
    print(f"Hoàn tất!")
    print(f"- Dòng hợp lệ: {success_count}")
    print(f"- Dòng bị loại bỏ: {error_count}")
    print(f"- Tỉ lệ lỗi: {error_count / (success_count + error_count) * 100:.2f}%")

# --- CẤU HÌNH ---
clean_tsv(
    input_path=r'D:\chuyen_nganh\Machine Translation version2\source\dataloader\EVBCorpus.tsv', 
    output_path=r'D:\chuyen_nganh\Machine Translation version2\source\dataloader\EVBCorpus_rs.tsv', 
    expected_columns=2  # Thay đổi số này tùy vào file của bạn (thường là 2 hoặc 3)
)