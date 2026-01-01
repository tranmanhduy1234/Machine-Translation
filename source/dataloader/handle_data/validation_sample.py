import random
def sample_tsv(input_path, output_path, k=5000, seed=42):
    random.seed(seed)
    reservoir = []

    with open(input_path, "r", encoding="utf-8") as f:
        header = f.readline()  # nếu có header
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = line

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(header)
        out.writelines(reservoir)

def first_k_lines(input_path, output_path, k=1000):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i >= k:
                break
            fout.write(line)    
            
from collections import deque

def last_k_lines(input_path, output_path, k=1000):
    print(f"Đang tìm {k} dòng cuối cùng...")
    with open(input_path, 'r', encoding='utf-8') as fin:
        # deque với maxlen=k sẽ tự động giữ lại k dòng cuối cùng
        last_lines = deque(fin, maxlen=k)
    
    print(f"Đang ghi vào file mới...")
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.writelines(last_lines)
    print("Hoàn thành!")
    
inputfile = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_RS_added.tsv"
# sample_tsv(inputfile, r"D:\chuyen_nganh\Dataset MT\sample1.tsv", k=5000)
first_k_lines(inputfile, r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\sample4.tsv", 1000)
# last_k_lines(inputfile, r"D:\chuyen_nganh\Dataset MT\sample3.tsv", 1000)