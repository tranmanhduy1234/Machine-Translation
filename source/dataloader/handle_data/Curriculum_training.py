import pandas as pd
import sentencepiece as spm 

sp = spm.SentencePieceProcessor()
sp.Load(model_file=r"D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model")
 
def safe_sort_tsv(input_path, output_path):
    print("bắt đầu xử lý...")
    print(f"Đọc file dữ liệu {input_path}")
    df = pd.read_csv(input_path, sep='\t', quoting=3, on_bad_lines="skip")
    
    print(f"Tính toán độ dài theo token với bộ mã hóa {sp.GetPieceSize()} token")
    df['temp_len'] = df['en'].astype(str).apply(lambda x: len(sp.EncodeAsIds(x)))
    
    print(f"Sắp xếp theo thứ tự tăng dần")
    df_sorted = df.sort_values(by="temp_len", ascending=True, kind="mergesort")
    
    print(f"Ghi dữ liệu mới")
    df_sorted.drop(columns=['temp_len']).to_csv(output_path, sep='\t', index=False, quoting=3)
    
    print("Hoan thanh")

input_file = r"D:\chuyen_nganh\Dataset MT\datasetTMD.tsv" 
output_file = r"D:\chuyen_nganh\Dataset MT\datasetTMD_sorted.tsv" 

safe_sort_tsv(input_file, output_file)