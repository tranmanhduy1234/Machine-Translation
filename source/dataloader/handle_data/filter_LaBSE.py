"""
Script lọc các câu có độ tương đồng nghĩa thấp sử dụng LaBSE
Xử lý file TSV lớn với batch processing
"""
import csv
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LaBSEFilter:
    def __init__(self, model_name='sentence-transformers/LaBSE', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Đang tải model LaBSE trên {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Model đã sẵn sàng!")
    
    def encode_batch(self, texts, batch_size=32, show_progress=True):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def calculate_similarity(self, emb1, emb2):
        if len(emb1.shape) == 1:
            return np.dot(emb1, emb2)
        return np.sum(emb1 * emb2, axis=1)
    
    def filter_tsv(self, 
                   input_file, 
                   output_file,
                   threshold=0.5,
                   src_col=0,
                   tgt_col=1,
                   batch_size=32,
                   chunk_size=10000,
                   sep='\t'):
        total_lines = 0
        kept_lines = 0
        
        print(f"\nBắt đầu lọc file: {input_file}")
        print(f"Ngưỡng similarity: {threshold}")
        print(f"Batch size: {batch_size}, Chunk size: {chunk_size}\n")
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for chunk_df in pd.read_csv(input_file, 
                                       sep=sep, 
                                       chunksize=chunk_size,
                                       header=None,
                                       on_bad_lines='skip'):
                
                src_texts = chunk_df.iloc[:, src_col].astype(str).tolist()
                tgt_texts = chunk_df.iloc[:, tgt_col].astype(str).tolist()
                
                print(f"Đang encode {len(src_texts)} cặp câu...")
                src_embeddings = self.encode_batch(src_texts, batch_size)
                tgt_embeddings = self.encode_batch(tgt_texts, batch_size)
                
                # Tính similarity
                similarities = self.calculate_similarity(src_embeddings, tgt_embeddings)
                
                # Lọc
                mask = similarities >= threshold
                filtered_df = chunk_df[mask]
                filtered_df = filtered_df.applymap(
                    lambda x: x.replace("\n", " ").replace("\r", " ").strip()
                    if isinstance(x, str) else x
                )
                # Lưu kết quả

                filtered_df.to_csv(
                    outf,
                    sep="\t",
                    header=False,
                    index=False,
                    mode="a",
                    lineterminator="\n",      # ép newline thống nhất
                    quoting=csv.QUOTE_NONE,    # không cho pandas tự thêm quote
                    escapechar="\\",           # escape nếu còn ký tự đặc biệt
                )
                
                total_lines += len(chunk_df)
                kept_lines += len(filtered_df)
                
                print(f"Chunk: {len(filtered_df)}/{len(chunk_df)} cặp đạt ngưỡng")
                print(f"Tổng cộng: {kept_lines}/{total_lines} ({kept_lines/total_lines*100:.2f}%)\n")
        
        print(f"\n✓ Hoàn thành!")
        print(f"Tổng số dòng: {total_lines}")
        print(f"Số dòng giữ lại: {kept_lines} ({kept_lines/total_lines*100:.2f}%)")
        print(f"Số dòng bỏ đi: {total_lines - kept_lines} ({(total_lines-kept_lines)/total_lines*100:.2f}%)")
        print(f"Kết quả đã lưu tại: {output_file}")
    
    def analyze_distribution(self,
                           input_file,
                           src_col=0,
                           tgt_col=1,
                           batch_size=32,
                           sample_size=None,
                           chunk_size=10000,
                           sep='\t'):
        print(f"\nPhân tích phân phối similarity trên {sample_size} mẫu...")
        return self._analyze_sample(input_file, src_col, tgt_col, 
                                    batch_size, sample_size, sep)
    
    def _analyze_sample(self, input_file, src_col, tgt_col, batch_size, sample_size, sep):
        """Phân tích trên mẫu"""
        # Đọc mẫu
        df_sample = pd.read_csv(input_file, 
                               sep=sep, 
                               nrows=sample_size,
                               header=None,
                               on_bad_lines='skip')
        
        src_texts = df_sample.iloc[:, src_col].astype(str).tolist()
        tgt_texts = df_sample.iloc[:, tgt_col].astype(str).tolist()
        
        src_embeddings = self.encode_batch(src_texts, batch_size)
        tgt_embeddings = self.encode_batch(tgt_texts, batch_size)
        similarities = self.calculate_similarity(src_embeddings, tgt_embeddings)
        
        self._print_statistics(similarities)
        return similarities
    
    def _print_statistics(self, similarities):
        """In thống kê similarity"""
        print(f"\n=== Thống kê Similarity ===")
        print(f"Số mẫu: {len(similarities):,}")
        print(f"Min: {similarities.min():.4f}")
        print(f"Max: {similarities.max():.4f}")
        print(f"Mean: {similarities.mean():.4f}")
        print(f"Median: {np.median(similarities):.4f}")
        print(f"Std: {similarities.std():.4f}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n=== Percentiles ===")
        for p in percentiles:
            val = np.percentile(similarities, p)
            print(f"P{p}: {val:.4f}")
        
        # Phân phối theo ngưỡng
        print(f"\n=== Phân phối theo ngưỡng ===")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for th in thresholds:
            count = (similarities >= th).sum()
            pct = count / len(similarities) * 100
            print(f"Threshold {th}: {count:,}/{len(similarities):,} ({pct:.2f}%)")

if __name__ == "__main__":
    filter_engine = LaBSEFilter()
    
    print("\n" + "="*50)
    print("BƯỚC 1: PHÂN TÍCH PHÂN PHỐI")
    print("="*50)
    
    similarities = filter_engine.analyze_distribution(
        input_file=r'D:\chuyen_nganh\Dataset MT\filtered_VietAI.tsv',  # Thay bằng file của bạn
        src_col=0,               
        tgt_col=1,                 
        sample_size=5000,           
        chunk_size=10000             
    )
    
    # 2. Lọc dữ liệu
    print("\n" + "="*50)
    print("BƯỚC 2: LỌC DỮ LIỆU")
    print("="*50)
    
    filter_engine.filter_tsv(
        input_file=r'D:\chuyen_nganh\Dataset MT\PhoMT.tsv',  
        output_file=r'D:\chuyen_nganh\Dataset MT\rs_filter_laBSE\PhoMT_rs.tsv',  
        threshold=0.8,                    
        src_col=0,                
        tgt_col=1,                 
        batch_size=24,                    
        chunk_size=100000               
    )
    
    print("\n✓ Xong! Dữ liệu đã được lọc.")