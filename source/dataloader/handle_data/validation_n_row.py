path = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD.tsv"
with open(path, 'r', encoding='utf-8') as f:
    num_lines = sum(1 for _ in f)
print(num_lines)