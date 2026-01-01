import re
import gzip
import json
from typing import Iterator, Tuple, Optional
from collections import Counter
import unicodedata

class EnhancedBitextProcessor:
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
        self.stats = {
            'total': 0,
            'passed': 0,
            'filtered_empty': 0,
            'filtered_alpha_ratio': 0,
            'filtered_length': 0,
            'filtered_length_ratio': 0,
            'filtered_number_mismatch': 0,
            'filtered_currency_mismatch': 0,
            'filtered_url_mismatch': 0,
            'filtered_email_mismatch': 0,
            'filtered_duplicate': 0,
            'filtered_special_chars': 0,
            'filtered_repetition': 0,
            'filtered_language_id': 0,
            'filtered_alignment_score': 0
        }
        self.seen_pairs = set()  # Để detect duplicates trong session
    
    def load_tsv_chunks(self, tsv_file: str) -> Iterator[list]:
        """Load dữ liệu từ file TSV (có thể nén .gz) theo chunk"""
        open_func = gzip.open if tsv_file.endswith('.gz') else open
        with open_func(tsv_file, 'rt', encoding='utf-8') as f:
            chunk = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    en, vi = parts[0], parts[1]
                    chunk.append((en, vi))
                    
                    if len(chunk) >= self.chunk_size:
                        yield chunk
                        chunk = []
            
            if chunk:
                yield chunk
    
    def deescape_special_chars(self, text: str) -> str:
        """Giải mã các ký tự đặc biệt HTML/XML"""
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
            '&nbsp;': ' ',
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Chuẩn hóa khoảng trắng"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def fix_quotes(self, text: str) -> str:
        """Chuẩn hóa dấu ngoặc kép"""
        text = re.sub(r'[""„‟]', '"', text)
        text = re.sub(r"[''‚‛]", "'", text)
        return text
    
    def remove_control_chars(self, text: str) -> str:
        return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\t\n\r')
    
    def filter_empty_lines(self, en: str, vi: str) -> bool:
        return bool(en.strip() and vi.strip())
    
    def filter_alpha_ratio(self, en: str, vi: str,
                          src_word_rat: float = 0.7,
                          trg_word_rat: float = 0.7,
                          src_alpha_rat: float = 0.7,
                          trg_alpha_rat: float = 0.7) -> bool:
        """Lọc dựa trên tỷ lệ ký tự chữ cái - CHẶT CHẼ HƠN"""
        
        def calc_alpha_ratio(text: str) -> float:
            if not text:
                return 0.0
            alpha_count = sum(c.isalpha() for c in text)
            return alpha_count / len(text)
        
        def calc_word_ratio(text: str) -> float:
            words = text.split()
            if not words:
                return 0.0
            alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
            return alpha_words / len(words)
        
        en_alpha = calc_alpha_ratio(en)
        vi_alpha = calc_alpha_ratio(vi)
        en_word = calc_word_ratio(en)
        vi_word = calc_word_ratio(vi)
        
        return (en_alpha >= src_alpha_rat and vi_alpha >= trg_alpha_rat and
                en_word >= src_word_rat and vi_word >= trg_word_rat)
    
    def filter_num_mismatch(self, en: str, vi: str, ratio: float = 0.7) -> bool:
        def extract_numbers(text: str) -> set:
            return set(re.findall(r'\d+(?:[.,]\d+)*', text))
        
        en_nums = extract_numbers(en)
        vi_nums = extract_numbers(vi)
        
        if not en_nums and not vi_nums:
            return True
        
        if not en_nums or not vi_nums:
            return len(en_nums) <= 1 and len(vi_nums) <= 1  # Cho phép 1 số khác biệt nhỏ
        
        # Chuẩn hóa dấu phân cách
        en_nums_normalized = {n.replace(',', '') for n in en_nums}
        vi_nums_normalized = {n.replace(',', '') for n in vi_nums}
        
        common = len(en_nums_normalized & vi_nums_normalized)
        total = max(len(en_nums_normalized), len(vi_nums_normalized))
        
        return (common / total) >= ratio
    
    def filter_currency_mismatch(self, en: str, vi: str) -> bool:
        """Lọc các cặp có sự khác biệt về ký hiệu tiền tệ"""
        currency_symbols = r'[$€£¥₹₽₫₩฿₱]'
        en_currencies = set(re.findall(currency_symbols, en))
        vi_currencies = set(re.findall(currency_symbols, vi))
        if not en_currencies and not vi_currencies:
            return True
        return en_currencies == vi_currencies
    
    def filter_url_mismatch(self, en: str, vi: str) -> bool:
        url_pattern = r'https?://[^\s]+'
        en_urls = set(re.findall(url_pattern, en))
        vi_urls = set(re.findall(url_pattern, vi))
        if en_urls or vi_urls:
            return len(en_urls.symmetric_difference(vi_urls)) <= 1
        
        return True
    
    def filter_email_mismatch(self, en: str, vi: str) -> bool:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        en_emails = set(re.findall(email_pattern, en))
        vi_emails = set(re.findall(email_pattern, vi))
        if not en_emails and not vi_emails:
            return True
        return en_emails == vi_emails
    
    def filter_duplicate(self, en: str, vi: str) -> bool:
        pair_hash = hash((en.lower(), vi.lower()))
        if pair_hash in self.seen_pairs:
            return False
        self.seen_pairs.add(pair_hash)
        return True
    
    def filter_special_chars_ratio(self, en: str, vi: str, max_ratio: float = 0.3) -> bool:
        def calc_special_ratio(text: str) -> float:
            if not text:
                return 0.0
            special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
            return special_count / len(text)
        en_special = calc_special_ratio(en)
        vi_special = calc_special_ratio(vi)
        
        return en_special <= max_ratio and vi_special <= max_ratio
    
    def filter_language_id(self, en: str, vi: str) -> bool:
        """Kiểm tra ngôn ngữ cơ bản - MỚI"""
        if not re.search(r'[a-zA-Z]', en):
            return False
        
        if not re.search(r'[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]', vi):
            return False
        
        vi_chars_in_en = len(re.findall(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', en.lower()))
        if vi_chars_in_en > len(en) * 0.05:
            return False
        return True
    
    def process_pair(self, en: str, vi: str) -> Optional[Tuple[str, str]]:
        if not self.filter_empty_lines(en, vi):
            self.stats['filtered_empty'] += 1
            return None
        
        en = self.remove_control_chars(en) # Bỏ các kí tự điều khiển
        vi = self.remove_control_chars(vi)
        en = self.deescape_special_chars(en) # Bỏ các kí tự đặc biệt HTML/XML
        vi = self.deescape_special_chars(vi)
        en = self.normalize_whitespace(en) # Chuẩn hóa khoảng trắng
        vi = self.normalize_whitespace(vi)
        en = self.fix_quotes(en) # Chuẩn hóa các dấu nháy
        vi = self.fix_quotes(vi)
        
        # Bước 3: Language ID - cơ bản
        if not self.filter_language_id(en, vi):
            self.stats['filtered_language_id'] += 1
            return None
        
        # Bước 4: Duplicate check - cơ bản
        if not self.filter_duplicate(en, vi):
            self.stats['filtered_duplicate'] += 1
            return None
        
        # # Bước 5: Alpha ratio - loại bỏ các câu có tỷ lệ kí tự chữ cái thấp
        if not self.filter_alpha_ratio(en, vi):
            self.stats['filtered_alpha_ratio'] += 1
            return None
        
        # # Bước 7: Special chars ratio - loại bỏ các câu có tỷ lệ kí tự đặc biệt cao > 0.3 theo mặc định
        if not self.filter_special_chars_ratio(en, vi):
            self.stats['filtered_special_chars'] += 1
            return None
        
        # # Bước 9: Number mismatch - loại các câu không tương đồng về con số.
        if not self.filter_num_mismatch(en, vi):
            self.stats['filtered_number_mismatch'] += 1
            return None
        
        # # Bước 10: Currency mismatch - loại các câu không tương đồng về đơn vị tiền tệ
        if not self.filter_currency_mismatch(en, vi):
            self.stats['filtered_currency_mismatch'] += 1
            return None
        
        # # Bước 11: URL mismatch - loại các câu không tương đồng về đường dẫn url
        if not self.filter_url_mismatch(en, vi):
            self.stats['filtered_url_mismatch'] += 1
            return None
        
        # # Bước 12: Email mismatch - loại các câu không tương đồng về email.
        if not self.filter_email_mismatch(en, vi):
            self.stats['filtered_email_mismatch'] += 1
            return None
        
        return (en, vi)
    
    def process_and_save(self, input_file: str, output_file: str):
        """Xử lý và lưu dữ liệu theo chunk"""
        
        print(f"Bắt đầu xử lý với Enhanced Filters")
        print(f"Chunk size: {self.chunk_size:,}")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print("=" * 70)
        
        chunk_num = 0
        open_func = gzip.open if output_file.endswith('.gz') else open
        
        with open_func(output_file, 'wt', encoding='utf-8') as f_out:
            for chunk in self.load_tsv_chunks(input_file):
                chunk_num += 1
                self.stats['total'] += len(chunk)
                
                for en, vi in chunk:
                    result = self.process_pair(en, vi)
                    
                    if result:
                        en_clean, vi_clean = result
                        f_out.write(f"{en_clean}\t{vi_clean}\n")
                        self.stats['passed'] += 1
                
                # In tiến độ
                if chunk_num % 10 == 0:
                    pass_rate = (self.stats['passed'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
                    print(f"Chunk {chunk_num:,} | Tổng: {self.stats['total']:,} | "
                          f"Giữ lại: {self.stats['passed']:,} ({pass_rate:.1f}%)")
    
    def print_stats(self):
        """In thống kê chi tiết"""
        print("\n" + "=" * 70)
        print("KẾT QUẢ XỬ LÝ CHI TIẾT")
        print("=" * 70)
        print(f"{'Tổng số cặp câu:':<35} {self.stats['total']:>15,}")
        print(f"{'Giữ lại:':<35} {self.stats['passed']:>15,}")
        print(f"{'Tổng lọc bỏ:':<35} {self.stats['total'] - self.stats['passed']:>15,}")
        
        if self.stats['total'] > 0:
            print(f"{'Tỷ lệ giữ lại:':<35} {self.stats['passed']/self.stats['total']*100:>14.2f}%")
        
        print("\n" + "-" * 70)
        print("CHI TIẾT CÁC BỘ LỌC:")
        print("-" * 70)
        
        filters = [
            ('Empty lines', 'filtered_empty'),
            ('Language ID', 'filtered_language_id'),
            ('Duplicate', 'filtered_duplicate'),
            ('Alpha ratio', 'filtered_alpha_ratio'),
            ('Length', 'filtered_length'),
            ('Length ratio', 'filtered_length_ratio'),
            ('Special chars ratio', 'filtered_special_chars'),
            ('Repetition', 'filtered_repetition'),
            ('Number mismatch', 'filtered_number_mismatch'),
            ('Currency mismatch', 'filtered_currency_mismatch'),
            ('URL mismatch', 'filtered_url_mismatch'),
            ('Email mismatch', 'filtered_email_mismatch'),
            ('Alignment score', 'filtered_alignment_score')
        ]
        
        for name, key in filters:
            count = self.stats[key]
            pct = (count / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
            print(f"{name:<35} {count:>10,} ({pct:>5.2f}%)")
        
        print("=" * 70)

# ==================== MAIN USAGE ====================

if __name__ == "__main__":
    # Khởi tạo processor
    processor = EnhancedBitextProcessor(chunk_size=100000)
    
    # Đường dẫn file
    input_file = r'D:\chuyen_nganh\Dataset MT\dataTrain.tsv'
    output_file = r'D:\chuyen_nganh\Dataset MT\dataTrain_rs.tsv'
    # Xử lý
    processor.process_and_save(input_file, output_file)
    
    # In thống kê
    processor.print_stats()
    print(f"\n✓ Đã lưu kết quả vào: {output_file}")
    print("Format: <English>\\t<Vietnamese>")