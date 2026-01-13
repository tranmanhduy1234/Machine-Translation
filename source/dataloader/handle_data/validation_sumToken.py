import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load(r"D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model")

def count_lines_and_tokens(file_path):
    total_tokens = 0
    valid_lines = 0
    skipped_lines = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                skipped_lines += 1
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                skipped_lines += 1
                continue

            src, tgt = parts
            total_tokens += len(sp.encode(src, out_type=int))
            total_tokens += len(sp.encode(tgt, out_type=int))
            valid_lines += 1

    return valid_lines, total_tokens, skipped_lines

if __name__ == "__main__":
    lines, tokens, skipped = count_lines_and_tokens(
        r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetEVBCorpus.tsv"
    )
    print(f"Số dòng hợp lệ: {lines}")
    print(f"Tổng số token: {tokens}")
    print(f"Dòng bị bỏ qua: {skipped}")