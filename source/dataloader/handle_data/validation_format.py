def check_tsv_format(file_path):
    total_lines = 0
    valid_pairs = 0
    error_lines = []

    print(f"Đang kiểm tra file: {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                
                # Bỏ qua dòng trống (nếu muốn tính cả dòng trống là lỗi thì xóa đoạn này)
                if not line:
                    continue

                parts = line.split('\t')

                # Kiểm tra xem có đúng 2 phần không
                if len(parts) == 2:
                    # Kiểm tra thêm xem nội dung có bị rỗng không
                    if parts[0].strip() and parts[1].strip():
                        valid_pairs += 1
                    else:
                        error_lines.append(f"Dòng {i}: Có tab nhưng một vế bị rỗng")
                else:
                    if len(parts) < 2:
                        error_lines.append(f"Dòng {i}: Không tìm thấy dấu tab nào")
                    else:
                        error_lines.append(f"Dòng {i}: Có nhiều hơn 1 dấu tab (thừa tab)")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu.")
        return

    print("-" * 30)
    print(f"✅ Tổng số dòng đã quét: {total_lines}")
    print(f"✅ Số cặp câu hợp lệ: {valid_pairs}")
    
    if len(error_lines) == 0:
        print("\n🎉 Tuyệt vời! File hoàn toàn đúng định dạng.")
    else:
        print(f"\n⚠️ Phát hiện {len(error_lines)} dòng lỗi:")
        # Chỉ in 10 lỗi đầu tiên để tránh tràn màn hình
        for err in error_lines[:10]:
            print(err)
        if len(error_lines) > 10:
            print(f"... và {len(error_lines) - 10} lỗi khác.")

# --- SỬ DỤNG ---
check_tsv_format(r'D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD.tsv') 