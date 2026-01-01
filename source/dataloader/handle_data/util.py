import pandas as pd
import os
import chardet
import gzip
import argparse
from pathlib import Path
import unicodedata
import re
from tqdm import tqdm

# kiểm tra định dạng csv ['en', 'vi']
def check_csv_columns(root_dir, required_columns=['en', 'vi']):
    results = []
    print(f"Bắt đầu quét từ thư mục: {root_dir}")
    print("-" * 40)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        csv_files = [f for f in filenames if f.endswith('.csv')]
        if not csv_files:
            continue
        print(f"Đang kiểm tra thư mục: {dirpath}")
        for filename in csv_files:
            file_path = os.path.join(dirpath, filename)
            try:
                df = pd.read_csv(file_path, nrows=0) 
                current_columns = list(df.columns)
                # Kiểm tra xem tất cả các cột bắt buộc có trong tệp không
                is_valid = all(col in current_columns for col in required_columns)
                # Lọc ra các cột còn thiếu
                missing_cols = [col for col in required_columns if col not in current_columns]
                results.append({
                    'file_path': file_path,
                    'valid': is_valid,
                    'current_columns': current_columns,
                    'missing_columns': missing_cols
                })
                status = "✅ Hợp lệ" if is_valid else "❌ KHÔNG HỢP LỆ"
                print(f"  [{status}] {filename}. Cột hiện tại: {current_columns}")
            except pd.errors.EmptyDataError:
                results.append({'file_path': file_path, 'valid': False, 'current_columns': [], 'missing_columns': required_columns})
                print(f"  [⚠️ Rỗng] {filename}. Tệp CSV rỗng hoặc chỉ có tiêu đề.")
            except Exception as e:
                results.append({'file_path': file_path, 'valid': False, 'current_columns': None, 'missing_columns': ['Error']})
                print(f"  [🚨 Lỗi] {filename}. Lỗi đọc tệp: {e}")
    print("-" * 40)
    # 3. Tổng hợp kết quả
    df_results = pd.DataFrame(results)
    print("\n## 📋 Tổng hợp kết quả kiểm tra")
    valid_count = df_results['valid'].sum()
    total_count = len(df_results)
    print(f"Tổng số tệp CSV được kiểm tra: {total_count}")
    print(f"Số tệp HỢP LỆ (có đủ cột 'en' và 'vi'): {valid_count}")
    print(f"Số tệp KHÔNG HỢP LỆ: {total_count - valid_count}")
    if not df_results.empty:
        # In các tệp không hợp lệ ra bảng
        invalid_files = df_results[df_results['valid'] == False]
        if not invalid_files.empty:
            print("\n### Tệp KHÔNG HỢP LỆ (Thiếu cột 'en' hoặc 'vi'):")
            print(invalid_files[['file_path', 'missing_columns']].to_string(index=False))

# check mã hóa
def check_encoding_of_valid_csvs(root_dir, required_columns=['en', 'vi']):
    print(f"Bắt đầu quét và kiểm tra mã hóa từ thư mục: {root_dir}")
    print("=" * 60)
    encoding_results = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    df_header = pd.read_csv(file_path, nrows=0)
                    current_columns = list(df_header.columns)
                    if all(col in current_columns for col in required_columns):
                        with open(file_path, 'rb') as f:
                            raw_data = f.read(1048576)
                        result = chardet.detect(raw_data)
                        encoding_results.append({
                            'file_path': file_path,
                            'encoding': result['encoding'],
                            'confidence': f"{result['confidence']:.2f}"
                        })

                        print(f"✅ [{result['encoding']} - {result['confidence']:.2f}] {file_path}")
                except pd.errors.EmptyDataError:
                    print(f"  [⚠️ Rỗng] {file_path}. Tệp CSV rỗng.")
                except Exception as e:
                    print(f"  [🚨 Lỗi] {file_path}. Lỗi: {e}")

# chia cắt file csv.
def chunk_file(input_file, n_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, "rb") as f:
        total_lines = sum(1 for _ in f) - 1
    lines_per_file = (total_lines + n_files - 1) // n_files
    try:
        with open(input_file, "rb") as f:
            header = f.readline()
            file_index = 1
            count = 0
            out = open(os.path.join(output_dir, f"data_part_{file_index:02d}.csv"), "wb")
            out.write(header)
            for line in f:
                if count >= lines_per_file and file_index < n_files:
                    out.close()
                    file_index += 1
                    out = open(os.path.join(output_dir, f"data_part_{file_index:02d}.csv"), "wb")
                    out.write(header)
                    count = 0
                out.write(line)
                count += 1
            out.close()
            print(f"Successfully split {input_file} into {file_index} files")
            print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        if 'out' in locals():
            out.close()

def convert_csv_to_moses(csv_file, output_prefix=None, chunk_size=10000, validate=True):
    """
    Chuyển đổi CSV sang định dạng MOSES với xử lý streaming
    Tối ưu cho dataset hàng trăm GB
    """
    if output_prefix is None:
        output_prefix = Path(csv_file).stem
    
    en_file = f"{output_prefix}.en.gz"
    vi_file = f"{output_prefix}.vi.gz"
    
    def clean_text(text):
        if pd.isna(text):
            return None
        text = str(text).strip()
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
        if not text or text.isspace():
            return None
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        return text if text else None
    
    total_rows = 0
    valid_rows = 0
    removed_count = 0
    
    try:
        # Đếm tổng dòng một cách hiệu quả
        print("📊 Đang đếm tổng dòng trong file...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  # Trừ header
        
        print(f"✓ Tổng dòng: {total_rows:,}")
        print("\n🔄 Bắt đầu xử lý...")
        
        # Xử lý chunk by chunk với streaming
        with gzip.open(en_file, 'wt', encoding='utf-8', compresslevel=6) as en_f, \
             gzip.open(vi_file, 'wt', encoding='utf-8', compresslevel=6) as vi_f:
            
            # Đọc từng chunk
            chunks_processed = 0
            for chunk in tqdm(
                pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False),
                total=(total_rows // chunk_size) + 1,
                desc="Xử lý chunks",
                unit="chunk"
            ):
                # Kiểm tra cột (chỉ làm lần đầu)
                if chunks_processed == 0:
                    if 'en' not in chunk.columns or 'vi' not in chunk.columns:
                        print("❌ File CSV phải chứa 2 cột 'en' và 'vi'")
                        return
                
                # Xử lý từng dòng trong chunk (streaming)
                for idx, row in chunk.iterrows():
                    en_clean = clean_text(row.get('en'))
                    vi_clean = clean_text(row.get('vi'))
                    
                    if en_clean and vi_clean:
                        en_f.write(en_clean + '\n')
                        vi_f.write(vi_clean + '\n')
                        valid_rows += 1
                    else:
                        removed_count += 1
                
                chunks_processed += 1
                
                # Giải phóng bộ nhớ chunk
                del chunk
        
        # Tính kích thước file output
        en_size = Path(en_file).stat().st_size / (1024**3)
        vi_size = Path(vi_file).stat().st_size / (1024**3)
        
        print("\n✅ Chuyển đổi thành công!")
        print(f"   📄 File tiếng Anh: {en_file} ({en_size:.2f} GB)")
        print(f"   📄 File tiếng Việt: {vi_file} ({vi_size:.2f} GB)")
        print(f"   📊 Dòng hợp lệ: {valid_rows:,}")
        print(f"   ❌ Dòng loại bỏ: {removed_count:,}")
        print(f"\n💡 Định dạng MOSES sẵn sàng cho OpusCleaner/bicleaner hardrules")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        # Xóa file lỗi
        for f in [en_file, vi_file]:
            if Path(f).exists():
                Path(f).unlink()

# Đầu vào là 2 file: en.gz và vi.gz (định dạng mose)
def convert_moses_to_csv(en_file, vi_file, output_file=None, chunk_size=10000):
    """
    Chuyển đổi MOSES sang CSV với xử lý streaming
    Tối ưu cho dataset hàng trăm GB
    """
    if output_file is None:
        output_file = "output.csv"
    
    try:
        # Đếm số dòng để hiển thị progress
        print("📊 Đang đếm số dòng...")
        with gzip.open(en_file, 'rt', encoding='utf-8') as f:
            en_count = sum(1 for line in f if line.strip())
        
        with gzip.open(vi_file, 'rt', encoding='utf-8') as f:
            vi_count = sum(1 for line in f if line.strip())
        
        if en_count != vi_count:
            print(f"⚠️ Cảnh báo: Số dòng không khớp!")
            print(f"  - File EN: {en_count:,} dòng")
            print(f"  - File VI: {vi_count:,} dòng")
            print(f"Sử dụng {min(en_count, vi_count):,} dòng chung")
        
        total_lines = min(en_count, vi_count)
        print(f"✓ Tổng dòng cần xử lý: {total_lines:,}")
        print("\n🔄 Bắt đầu chuyển đổi...")
        
        # Mở cả 2 file đồng thời và ghi streaming
        with gzip.open(en_file, 'rt', encoding='utf-8') as en_f, \
             gzip.open(vi_file, 'rt', encoding='utf-8') as vi_f, \
             open(output_file, 'w', encoding='utf-8', newline='') as csv_f:
            
            # Ghi header
            csv_f.write('en,vi\n')
            
            # Xử lý từng batch
            batch = []
            lines_written = 0
            
            with tqdm(total=total_lines, desc="Ghi CSV", unit="dòng") as pbar:
                for en_line, vi_line in zip(en_f, vi_f):
                    en_text = en_line.strip()
                    vi_text = vi_line.strip()
                    
                    if not en_text or not vi_text:
                        continue
                    
                    # Escape quotes và commas theo chuẩn CSV
                    en_text = en_text.replace('"', '""')
                    vi_text = vi_text.replace('"', '""')
                    
                    # Thêm quotes nếu có dấu phẩy hoặc newline
                    if ',' in en_text or '\n' in en_text:
                        en_text = f'"{en_text}"'
                    if ',' in vi_text or '\n' in vi_text:
                        vi_text = f'"{vi_text}"'
                    
                    batch.append(f'{en_text},{vi_text}\n')
                    lines_written += 1
                    
                    # Ghi batch khi đủ kích thước
                    if len(batch) >= chunk_size:
                        csv_f.writelines(batch)
                        pbar.update(len(batch))
                        batch = []
                    
                    # Dừng khi đạt total_lines
                    if lines_written >= total_lines:
                        break
                
                # Ghi batch còn lại
                if batch:
                    csv_f.writelines(batch)
                    pbar.update(len(batch))
        
        # Tính kích thước file output
        output_size = Path(output_file).stat().st_size / (1024**3)
        
        print(f"\n✅ Chuyển đổi thành công!")
        print(f"   📄 File đầu ra: {output_file} ({output_size:.2f} GB)")
        print(f"   📊 Số dòng: {lines_written:,}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

# Hàm tiện ích để xử lý file cực lớn với buffer tối ưu
def estimate_memory_usage(csv_file, sample_size=1000):
    """
    Ước tính việc sử dụng RAM và đề xuất chunk_size phù hợp
    """
    try:
        # Đọc mẫu để ước tính
        sample = pd.read_csv(csv_file, nrows=sample_size)
        memory_per_row = sample.memory_usage(deep=True).sum() / len(sample)
        
        print(f"📊 Phân tích file:")
        print(f"   - Bộ nhớ trung bình/dòng: {memory_per_row:.2f} bytes")
        
        # Ước tính chunk_size an toàn (giữ dưới 500MB RAM)
        target_memory_mb = 500
        suggested_chunk_size = int((target_memory_mb * 1024 * 1024) / memory_per_row)
        
        print(f"   - Chunk size đề xuất: {suggested_chunk_size:,} dòng")
        print(f"   - RAM ước tính mỗi chunk: ~{target_memory_mb}MB")
        
        return suggested_chunk_size
    except Exception as e:
        print(f"⚠️ Không thể ước tính: {e}")
        return 10000  # Giá trị mặc định an toàn

# Đầu vào là 2 file: en.gz và vi.gz (định dạng mose)
def validate_moses_files(en_file, vi_file, chunk_size=50000):
    """
    Xác thực TOÀN BỘ tập dữ liệu MOSES (40GB+) bằng chunk processing
    
    Args:
        en_file: File tiếng Anh (.en.gz)
        vi_file: File tiếng Việt (.vi.gz)
        chunk_size: Số dòng xử lý 1 lần (50000 mặc định)
    
    Returns:
        dict: Kết quả validation chi tiết
    """
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    en_path = Path(en_file)
    vi_path = Path(vi_file)
    
    # 1. Kiểm tra file tồn tại
    print("📋 Bắt đầu xác thực TOÀN BỘ file MOSES...")
    print("-" * 70)
    
    if not en_path.exists() or not vi_path.exists():
        results['errors'].append("File không tồn tại")
        results['valid'] = False
        return results
    
    print(f"✓ File tồn tại")
    print(f"  EN: {en_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  VI: {vi_path.stat().st_size / (1024**3):.2f} GB")
    
    # 2. Kiểm tra toàn bộ dữ liệu chunk by chunk
    print("\n🔍 Kiểm tra TOÀN BỘ dữ liệu...")
    
    total_lines = 0
    mismatches = []
    line_errors = []
    
    # Thống kê
    en_stats = {
        'empty': 0,
        'whitespace_only': 0,
        'too_short': 0,  # < 3 chars
        'too_long': 0,   # > 10000 chars
        'min_length': float('inf'),
        'max_length': 0,
        'total_length': 0,
        'with_numbers': 0,
        'with_special_chars': 0,
        'lines_processed': 0,
    }
    
    vi_stats = {
        'empty': 0,
        'whitespace_only': 0,
        'too_short': 0,
        'too_long': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'total_length': 0,
        'with_numbers': 0,
        'with_special_chars': 0,
        'lines_processed': 0,
    }
    
    try:
        with gzip.open(en_file, 'rt', encoding='utf-8', errors='strict') as en_f, \
             gzip.open(vi_file, 'rt', encoding='utf-8', errors='strict') as vi_f:
            
            en_buffer = []
            vi_buffer = []
            chunk_count = 0
            
            # Đọc chunk by chunk
            for en_line in en_f:
                en_buffer.append(en_line.rstrip('\n'))
                
                if len(en_buffer) >= chunk_size:
                    # Đọc chunk VI tương ứng
                    for _ in range(chunk_size):
                        vi_line = vi_f.readline()
                        if vi_line:
                            vi_buffer.append(vi_line.rstrip('\n'))
                    
                    # Kiểm tra chunk
                    if len(en_buffer) != len(vi_buffer):
                        mismatches.append({
                            'chunk': chunk_count,
                            'en_count': len(en_buffer),
                            'vi_count': len(vi_buffer)
                        })
                    
                    # Xử lý chunk
                    for idx, (en_text, vi_text) in enumerate(zip(en_buffer, vi_buffer)):
                        global_idx = chunk_count * chunk_size + idx
                        
                        # Kiểm tra EN
                        if not en_text:
                            en_stats['empty'] += 1
                            line_errors.append((global_idx, 'EN_empty'))
                        elif en_text.isspace():
                            en_stats['whitespace_only'] += 1
                            line_errors.append((global_idx, 'EN_whitespace'))
                        else:
                            en_len = len(en_text)
                            en_stats['min_length'] = min(en_stats['min_length'], en_len)
                            en_stats['max_length'] = max(en_stats['max_length'], en_len)
                            en_stats['total_length'] += en_len
                            
                            if en_len < 3:
                                en_stats['too_short'] += 1
                            if en_len > 10000:
                                en_stats['too_long'] += 1
                            
                            if any(c.isdigit() for c in en_text):
                                en_stats['with_numbers'] += 1
                            if any(not c.isalnum() and not c.isspace() for c in en_text):
                                en_stats['with_special_chars'] += 1
                        
                        en_stats['lines_processed'] += 1
                        
                        # Kiểm tra VI
                        if not vi_text:
                            vi_stats['empty'] += 1
                            line_errors.append((global_idx, 'VI_empty'))
                        elif vi_text.isspace():
                            vi_stats['whitespace_only'] += 1
                            line_errors.append((global_idx, 'VI_whitespace'))
                        else:
                            vi_len = len(vi_text)
                            vi_stats['min_length'] = min(vi_stats['min_length'], vi_len)
                            vi_stats['max_length'] = max(vi_stats['max_length'], vi_len)
                            vi_stats['total_length'] += vi_len
                            
                            if vi_len < 3:
                                vi_stats['too_short'] += 1
                            if vi_len > 10000:
                                vi_stats['too_long'] += 1
                            
                            if any(c.isdigit() for c in vi_text):
                                vi_stats['with_numbers'] += 1
                            if any(not c.isalnum() and not c.isspace() for c in vi_text):
                                vi_stats['with_special_chars'] += 1
                        
                        vi_stats['lines_processed'] += 1
                    
                    total_lines += len(en_buffer)
                    chunk_count += 1
                    
                    # Progress
                    if chunk_count % 10 == 0:
                        print(f"  ✓ Đã xử lý {total_lines:,} dòng ({chunk_count} chunks)...")
                    
                    en_buffer = []
                    vi_buffer = []
            
            # Xử lý chunk cuối cùng (nếu có)
            if en_buffer:
                for _ in range(len(en_buffer)):
                    vi_line = vi_f.readline()
                    if vi_line:
                        vi_buffer.append(vi_line.rstrip('\n'))
                
                for idx, (en_text, vi_text) in enumerate(zip(en_buffer, vi_buffer)):
                    global_idx = chunk_count * chunk_size + idx
                    
                    if not en_text or en_text.isspace():
                        en_stats['empty'] += 1
                    else:
                        en_len = len(en_text)
                        en_stats['min_length'] = min(en_stats['min_length'], en_len)
                        en_stats['max_length'] = max(en_stats['max_length'], en_len)
                        en_stats['total_length'] += en_len
                    
                    if not vi_text or vi_text.isspace():
                        vi_stats['empty'] += 1
                    else:
                        vi_len = len(vi_text)
                        vi_stats['min_length'] = min(vi_stats['min_length'], vi_len)
                        vi_stats['max_length'] = max(vi_stats['max_length'], vi_len)
                        vi_stats['total_length'] += vi_len
                
                total_lines += len(en_buffer)
    
    except UnicodeDecodeError as e:
        results['errors'].append(f"Lỗi encoding: {e}")
        results['valid'] = False
        return results
    
    except Exception as e:
        results['errors'].append(f"Lỗi: {e}")
        results['valid'] = False
        return results
    
    # Tính trung bình
    en_avg = en_stats['total_length'] / max(en_stats['lines_processed'] - en_stats['empty'], 1)
    vi_avg = vi_stats['total_length'] / max(vi_stats['lines_processed'] - vi_stats['empty'], 1)
    
    if en_stats['min_length'] == float('inf'):
        en_stats['min_length'] = 0
    if vi_stats['min_length'] == float('inf'):
        vi_stats['min_length'] = 0
    
    # In kết quả
    print(f"\n✅ Hoàn thành quét TOÀN BỘ {total_lines:,} dòng")
    print("\n" + "=" * 70)
    print("📊 THỐNG KÊ TIẾNG ANH:")
    print(f"   Tổng dòng: {en_stats['lines_processed']:,}")
    print(f"   Dòng trống: {en_stats['empty']:,}")
    print(f"   Dòng chỉ khoảng trắng: {en_stats['whitespace_only']:,}")
    print(f"   Dòng quá ngắn (<3 chars): {en_stats['too_short']:,}")
    print(f"   Dòng quá dài (>10000 chars): {en_stats['too_long']:,}")
    print(f"   Min/Max/Avg length: {en_stats['min_length']}/{en_stats['max_length']}/{en_avg:.1f}")
    print(f"   Có số: {en_stats['with_numbers']:,}")
    print(f"   Có ký tự đặc biệt: {en_stats['with_special_chars']:,}")
    
    print("\n📊 THỐNG KÊ TIẾNG VIỆT:")
    print(f"   Tổng dòng: {vi_stats['lines_processed']:,}")
    print(f"   Dòng trống: {vi_stats['empty']:,}")
    print(f"   Dòng chỉ khoảng trắng: {vi_stats['whitespace_only']:,}")
    print(f"   Dòng quá ngắn (<3 chars): {vi_stats['too_short']:,}")
    print(f"   Dòng quá dài (>10000 chars): {vi_stats['too_long']:,}")
    print(f"   Min/Max/Avg length: {vi_stats['min_length']}/{vi_stats['max_length']}/{vi_avg:.1f}")
    print(f"   Có số: {vi_stats['with_numbers']:,}")
    print(f"   Có ký tự đặc biệt: {vi_stats['with_special_chars']:,}")
    
    # Cảnh báo
    print("\n" + "=" * 70)
    if en_stats['empty'] > 0 or vi_stats['empty'] > 0:
        results['warnings'].append(f"Có dòng trống: EN={en_stats['empty']:,}, VI={vi_stats['empty']:,}")
        results['valid'] = False
    
    if en_stats['too_short'] > 0 or vi_stats['too_short'] > 0:
        results['warnings'].append(f"Có dòng quá ngắn: EN={en_stats['too_short']:,}, VI={vi_stats['too_short']:,}")
    
    if en_stats['too_long'] > 0 or vi_stats['too_long'] > 0:
        results['warnings'].append(f"Có dòng quá dài: EN={en_stats['too_long']:,}, VI={vi_stats['too_long']:,}")
    
    if mismatches:
        results['errors'].append(f"Số dòng không khớp ở {len(mismatches)} chunk(s)")
        results['valid'] = False
    
    if results['valid'] and not results['errors']:
        print("✅ XÁC THỰC THÀNH CÔNG!")
        print("   Dữ liệu sẵn sàng cho training dịch máy")
    else:
        print("❌ XÁC THỰC CÓ VẤN ĐỀ!")
        for err in results['errors']:
            print(f"   ❌ {err}")
    
    if results['warnings']:
        for warn in results['warnings']:
            print(f"   ⚠️  {warn}")
    
    print("=" * 70)
    
    results['stats'] = {
        'total_lines': total_lines,
        'en': en_stats,
        'vi': vi_stats,
        'chunk_size': chunk_size,
    }
    
    return results

def concatTSV():
    dataTrain = r"D:\chuyen_nganh\Dataset MT\dataTrain.tsv"
    root = r"D:\chuyen_nganh\Dataset MT"
    with open(dataTrain, 'w', encoding='utf-8') as fout:
        fout.write("en	vi\n")
        for file_name in sorted(os.listdir(root)):
            if not file_name.endswith(".tsv"):
                continue
            if file_name == "dataTrain.tsv":
                continue
            path = os.path.join(root, file_name)
            with open(path, 'r', encoding='utf-8') as fin:
                for i, line in enumerate(fin):
                    fout.write(line)
    
if __name__=="__main__":
    concatTSV()