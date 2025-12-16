"""
    BẢN DEMO
"""
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# Download required data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ============ 1. BLEU SCORE CÓ LEVEL (BLEU-1, BLEU-2, BLEU-3, BLEU-4) ============

def calculate_bleu_with_levels(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Tính BLEU score với các level khác nhau
    
    Args:
        reference: câu tham chiếu (string)
        hypothesis: câu dự đoán (string)
        weights: trọng số cho 1-gram, 2-gram, 3-gram, 4-gram
    
    Returns:
        dict: BLEU-1, BLEU-2, BLEU-3, BLEU-4, BLEU-4 tổng hợp
    """
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    
    # Chuyển thành danh sách tham chiếu
    reference_tokens = [ref_tokens]
    
    smoothing = SmoothingFunction().method1
    
    results = {
        'BLEU-1': sentence_bleu(reference_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        'BLEU-2': sentence_bleu(reference_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        'BLEU-3': sentence_bleu(reference_tokens, hyp_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing),
        'BLEU-4': sentence_bleu(reference_tokens, hyp_tokens, weights=weights, smoothing_function=smoothing),
    }
    
    return results


# ============ 2. TÍNH BLEU CHO CẢ CORPUS (NHIỀU CẶP CÂU) ============

def calculate_corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Tính BLEU score trên toàn bộ corpus
    
    Args:
        references: danh sách câu tham chiếu (list of strings)
        hypotheses: danh sách câu dự đoán (list of strings)
        weights: trọng số cho n-grams
    
    Returns:
        float: BLEU score (0-1)
    """
    ref_tokens_list = [[word_tokenize(ref.lower())] for ref in references]
    hyp_tokens_list = [word_tokenize(hyp.lower()) for hyp in hypotheses]
    
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(ref_tokens_list, hyp_tokens_list, weights=weights, smoothing_function=smoothing)
    
    return bleu_score


# ============ 3. NHIỀU THAM CHIẾU (MULTIPLE REFERENCES) ============

def calculate_bleu_multiple_refs(references, hypothesis):
    """
    Tính BLEU khi có nhiều bản dịch tham chiếu
    
    Args:
        references: danh sách bản dịch tham chiếu (list of strings)
        hypothesis: câu dự đoán (string)
    
    Returns:
        float: BLEU-4 score
    """
    ref_tokens_list = [word_tokenize(ref.lower()) for ref in references]
    hyp_tokens = word_tokenize(hypothesis.lower())
    
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(ref_tokens_list, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return bleu_score


# ============ 4. VÍ DỤ THỰC HÀNH ============

if __name__ == "__main__":
    # Ví dụ 1: Một cặp câu
    print("=" * 60)
    print("VÍ DỤ 1: Một cặp câu dịch")
    print("=" * 60)
    
    ref = "The cat is on the table"
    hyp = "The cat is on a table"
    
    bleu_scores = calculate_bleu_with_levels(ref, hyp)
    print(f"Tham chiếu: {ref}")
    print(f"Dự đoán:    {hyp}")
    for level, score in bleu_scores.items():
        print(f"{level}: {score:.4f}")
    
    # Ví dụ 2: Nhiều cặp câu (Corpus)
    print("\n" + "=" * 60)
    print("VÍ DỤ 2: Nhiều cặp câu (Corpus)")
    print("=" * 60)
    
    references = [
        "The cat is on the table",
        "I love machine translation",
        "Hello world"
    ]
    
    hypotheses = [
        "The cat is on a table",
        "I like machine translation",
        "Hello world"
    ]
    
    corpus_score = calculate_corpus_bleu(references, hypotheses)
    print(f"BLEU Score (Corpus): {corpus_score:.4f}")
    
    # Ví dụ 3: Nhiều tham chiếu
    print("\n" + "=" * 60)
    print("VÍ DỤ 3: Nhiều bản dịch tham chiếu")
    print("=" * 60)
    
    refs = [
        "The cat is on the table",
        "There is a cat on the table",
        "A cat sits on the table"
    ]
    hyp = "The cat is on a table"
    
    multi_ref_score = calculate_bleu_multiple_refs(refs, hyp)
    print(f"Tham chiếu 1: {refs[0]}")
    print(f"Tham chiếu 2: {refs[1]}")
    print(f"Tham chiếu 3: {refs[2]}")
    print(f"Dự đoán:      {hyp}")
    print(f"BLEU-4 (Multiple Refs): {multi_ref_score:.4f}")