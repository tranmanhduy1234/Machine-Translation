import re

def clean_text_file(input_path, output_path):
    vietnamese_chars = "a-zA-Z0-9áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ"
    # Giữ lại chữ cái, số, dấu cách và các dấu câu cơ bản (. , ? ! : ;)
    pattern = re.compile(f"[^{vietnamese_chars}\s\.,\?!\:;]")
    print(f"Đang bắt đầu lọc file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(fin):
            line = line.replace('\t', '\n')
            clean_line = pattern.sub('', line)
            fout.write(clean_line)
            
            if i % 1000000 == 0 and i > 0:
                print(f"Đã xử lý {i // 1000000} triệu dòng...")

    print(f"Xong! File sạch đã được lưu tại: {output_path}")
    
# Sử dụng
clean_text_file(r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\tokenization.txt',
                r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\tokenization_clean.txt')