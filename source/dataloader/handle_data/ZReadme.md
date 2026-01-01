---
license: unknown
task_categories:
- translation
language:
- en
- vi
pretty_name: Machine Translation Dataset 2025
size_categories:
- 10M<n<100M
---
Dữ liệu được thu thập từ nhiều nguồn:
- CCMatrix: https://opus.nlpl.eu/CCMatrix/en&vi/v1/CCMatrix
- OpenSubtitles: https://opus.nlpl.eu/OpenSubtitles/en&vi/v2024/OpenSubtitles
- MultiHPLT: https://opus.nlpl.eu/MultiHPLT/en&vi/v2/MultiHPLT
- CCAligned: https://opus.nlpl.eu/CCAligned/en&vi/v1/CCAligned
- ParaCrawl: https://opus.nlpl.eu/ParaCrawl-Bonus/en&vi/v9/ParaCrawl-Bonus
- PhoMT: https://huggingface.co/datasets/ura-hcmut/PhoMT
- VietAI: https://huggingface.co/datasets/wanhin/VietAI_MTet

Dữ liệu đã trải qua quy trình lọc cơ bản  
- Alpha_ratio: tỷ lệ chữ cái/tổng số kí tự và tỷ lệ tổng số từ/tổng lượng từ (cả 2 vế src và tgt) - 0.7
- Deescape_special_chars & deescape_tsv: chuẩn hóa html empty thành cách kí tự thông thường.
- Normalize_whitespace: chuẩn hóa khoảng trắng liên tiếp về một khoảng trắng.
- Remove_empty_line
- Currency_mismatch: loại bỏ dòng sai xót đơn vị tiền tệ
- Num_mismatch: loại bỏ dòng sai về đơn vị số
- Remove_control_char
- Deduplicate: loại bỏ trùng lặp sử dụng mã hash
- Url_mismatch: loại bỏ dòng sai lệch địa chỉ url
- email_mismatch: loại bỏ dòng sai lệch về email

Lọc nâng cao  
- Sử dụng model fasttext lọc các dòng sai về ngôn ngữ
- Sử dụng model laBSE lọc đi các cặp câu không tương đồng ý nghĩa > 0.8


Bộ dữ liệu bao gồm <b>35,628,206</b> cặp câu song ngữ Anh-Việt.  
Tổng số lượng token: <b>2,189,438,317</b> 
Độ dài mỗi câu >= 5 và <= 250 token. Sử dụng sentencepiece-unigram tạo bộ từ điển      
Lưu ý: Bộ dữ liệu được sắp xếp theo thứ tự tăng dần về độ dài câu nguồn.