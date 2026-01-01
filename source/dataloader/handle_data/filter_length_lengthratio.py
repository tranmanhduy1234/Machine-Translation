import sentencepiece as spm
from tqdm import tqdm

def filterLength(
    filepath,
    sp_model,
    output_path,
    batch_size=1000000,
    min_length=1,
    max_length=250,
    ratio_threshold=1.35
):
    print(f"Đang xử lý file: {filepath}")
    print(f"Tham số lọc:")
    print(f"  - Độ dài min: {min_length}")
    print(f"  - Độ dài max: {max_length}")
    print(f"  - Ngưỡng tỷ lệ: {ratio_threshold}")
    
    stats = {
        'total_lines': 0,
        'invalid_format': 0,
        'empty_lines': 0,
        'too_short': 0,
        'too_long': 0,
        'zero_length': 0,
        'filtered_by_ratio': 0,
        'kept_lines': 0,
        'total_src_tokens': 0,
        'total_tgt_tokens': 0,
        'max_ratio': 0.0,
        'min_ratio': float('inf'),
        'max_src_length': 0,
        'max_tgt_length': 0
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    out_f = open(output_path, 'w', encoding='utf-8')
    source_batch = []
    target_batch = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            stats['total_lines'] += 1
            
            line = line.strip()
            if not line:
                stats['empty_lines'] += 1
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                stats['invalid_format'] += 1
                continue
            
            source_batch.append(parts[0])
            target_batch.append(parts[1])
            
            if len(source_batch) >= batch_size:
                _process_batch(
                    source_batch,
                    target_batch,
                    sp_model,
                    out_f,
                    min_length,
                    max_length,
                    ratio_threshold,
                    stats
                )
                source_batch.clear()
                target_batch.clear()
        
        if source_batch:
            _process_batch(
                source_batch,
                target_batch,
                sp_model,
                out_f,
                min_length,
                max_length,
                ratio_threshold,
                stats
            )
    
    out_f.close()
    
    _print_statistics(stats, output_path)
    
    return stats

def _process_batch(src_batch, tgt_batch, sp_model, out_f, min_len, max_len, ratio_threshold, stats):
    """Xử lý một batch dữ liệu"""
    src_ids = sp_model.EncodeAsIds(src_batch)
    tgt_ids = sp_model.EncodeAsIds(tgt_batch)
    
    for src, tgt, s_ids, t_ids in zip(src_batch, tgt_batch, src_ids, tgt_ids):
        len_s = len(s_ids)
        len_t = len(t_ids)
        
        if len_s == 0 or len_t == 0:
            stats['zero_length'] += 1
            continue
        
        if len_s < min_len or len_t < min_len:
            stats['too_short'] += 1
            continue
        
        if len_s > max_len or len_t > max_len:
            stats['too_long'] += 1
            continue
        
        ratio = max(len_s, len_t) / min(len_s, len_t)
        
        stats['max_ratio'] = max(stats['max_ratio'], ratio)
        stats['min_ratio'] = min(stats['min_ratio'], ratio)
        
        stats['max_src_length'] = max(stats['max_src_length'], len_s)
        stats['max_tgt_length'] = max(stats['max_tgt_length'], len_t)
        
        if ratio >= ratio_threshold:
            stats['filtered_by_ratio'] += 1
            continue
        
        out_f.write(f"{src}\t{tgt}\n")
        stats['kept_lines'] += 1
        stats['total_src_tokens'] += len_s
        stats['total_tgt_tokens'] += len_t

def _print_statistics(stats, output_path):
    """In thống kê kết quả"""
    print("\n" + "="*70)
    print("THỐNG KÊ KẾT QUẢ LỌC DỮ LIỆU")
    print("="*70)
    
    print("\nTHỐNG KÊ ĐẦU VÀO:")
    print(f"  Tổng số dòng:                    {stats['total_lines']:>15,}")
    
    print("\nTHỐNG KÊ LỌC:")
    print(f"  Dòng trống:                      {stats['empty_lines']:>15,}")
    print(f"  Sai định dạng:                   {stats['invalid_format']:>15,}")
    print(f"  Độ dài = 0:                      {stats['zero_length']:>15,}")
    print(f"  Quá ngắn (< min_length):         {stats['too_short']:>15,}")
    print(f"  Quá dài (> max_length):          {stats['too_long']:>15,}")
    print(f"  Tỷ lệ độ dài vượt ngưỡng:        {stats['filtered_by_ratio']:>15,}")
    
    total_filtered = (stats['empty_lines'] + stats['invalid_format'] + 
                     stats['zero_length'] + stats['too_short'] + 
                     stats['too_long'] + stats['filtered_by_ratio'])
    print(f"  Tổng số dòng bị lọc:             {total_filtered:>15,}")
    
    print("THỐNG KÊ ĐẦU RA:")
    print(f"  Số dòng giữ lại:                 {stats['kept_lines']:>15,}")
    
    if stats['total_lines'] > 0:
        keep_rate = (stats['kept_lines'] / stats['total_lines']) * 100
        filter_rate = (total_filtered / stats['total_lines']) * 100
        print(f"  Tỷ lệ giữ lại:                   {keep_rate:>14.2f}%")
        print(f"  Tỷ lệ lọc:                       {filter_rate:>14.2f}%")
    
    if stats['kept_lines'] > 0:
        avg_src = stats['total_src_tokens'] / stats['kept_lines']
        avg_tgt = stats['total_tgt_tokens'] / stats['kept_lines']
        print("\n THỐNG KÊ ĐỘ DÀI (tokens):")
        print(f"  TB tokens nguồn:                 {avg_src:>14.2f}")
        print(f"  TB tokens đích:                  {avg_tgt:>14.2f}")
        print(f"  Max tokens nguồn:                {stats['max_src_length']:>15,}")
        print(f"  Max tokens đích:                 {stats['max_tgt_length']:>15,}")
    
    if stats['min_ratio'] != float('inf'):
        print("\nTHỐNG KÊ TỶ LỆ ĐỘ DÀI:")
        print(f"  Tỷ lệ nhỏ nhất:                  {stats['min_ratio']:>14.2f}")
        print(f"  Tỷ lệ lớn nhất:                  {stats['max_ratio']:>14.2f}")
    
    print("\n" + "="*70)
    print(f"Đã ghi file: {output_path}")
    print("="*70 + "\n")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.Load(r"D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model")
    
    # Lọc dữ liệu
    stats = filterLength(
        filepath=r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD.tsv",
        sp_model=sp,
        output_path="filtered_VietAIv2.tsv",
        batch_size=1000000,
        min_length=5,    
        max_length=250,      
        ratio_threshold=1.3  
    )