import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import re
import string
from collections import Counter

# Đảm bảo đã tải NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class VQAEvaluator:
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        self.metrics = config['evaluation']['metrics']
        
    def compute_metrics(self, predictions, references):
        """Tính toán các metric dựa trên predictions và references"""
        results = {}
        
        # Loại bỏ các ký tự đặc biệt và chuẩn hóa text
        normalized_predictions = [self._normalize_text(pred) for pred in predictions]
        normalized_references = [self._normalize_text(ref) for ref in references]
        
        # Tính các metric đã được cấu hình
        if 'accuracy' in self.metrics:
            results['accuracy'] = self._compute_accuracy(normalized_predictions, normalized_references)
        
        if 'f1' in self.metrics:
            results['f1'] = self._compute_f1(normalized_predictions, normalized_references)
        
        if 'bleu' in self.metrics:
            results['bleu'] = self._compute_bleu(normalized_predictions, normalized_references)
        
        return results
    
    def _normalize_text(self, text):
        """Chuẩn hóa văn bản để đánh giá"""
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ dấu câu
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _compute_accuracy(self, predictions, references):
        """Tính độ chính xác dựa trên match chính xác"""
        correct = 0
        for pred, ref in zip(predictions, references):
            if pred == ref:
                correct += 1
        
        return correct / len(predictions) if predictions else 0
    
    def _compute_f1(self, predictions, references):
        """Tính F1 score dựa trên token-level"""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(nltk.word_tokenize(pred))
            ref_tokens = set(nltk.word_tokenize(ref))
            
            # Tính precision, recall và F1
            common_tokens = pred_tokens.intersection(ref_tokens)
            
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _compute_bleu(self, predictions, references):
        """Tính BLEU score"""
        smoothie = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred)
            ref_tokens = [nltk.word_tokenize(ref)]
            
            # Tính BLEU score
            try:
                bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)
            except Exception as e:
                # Xử lý trường hợp đặc biệt (ví dụ: câu rỗng)
                bleu_scores.append(0)
        
        return np.mean(bleu_scores)
