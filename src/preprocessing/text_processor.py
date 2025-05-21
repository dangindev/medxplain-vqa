import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
import torch

# Đảm bảo đã tải NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.max_question_length = config['preprocessing']['text']['max_question_length']
        self.max_answer_length = config['preprocessing']['text']['max_answer_length']
        
        # Sử dụng simple tokenizer thay vì transformers
        print("Sử dụng NLTK tokenizer thay vì transformer tokenizer")
        self.use_transformer_tokenizer = False
        
        # Tạo thư mục processed nếu chưa tồn tại
        self.processed_dir = Path(config['data']['processed_dir']) / "text"
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def clean_text(self, text):
        """Làm sạch văn bản"""
        # Chuyển đổi về chữ thường
        text = text.lower()
        
        # Loại bỏ ký tự đặc biệt và số
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize văn bản sử dụng NLTK"""
        return word_tokenize(text)
    
    def process_question(self, question):
        """Xử lý câu hỏi"""
        cleaned_question = self.clean_text(question)
        return cleaned_question
    
    def process_answer(self, answer):
        """Xử lý câu trả lời"""
        cleaned_answer = self.clean_text(answer)
        return cleaned_answer
    
    def encode_question(self, question):
        """Mã hóa câu hỏi"""
        if self.use_transformer_tokenizer:
            # Đoạn code này không được gọi vì chúng ta không sử dụng transformer tokenizer
            pass
        else:
            # Sử dụng NLTK tokenize đơn giản
            tokens = self.tokenize_text(question)
            return tokens
    
    def encode_answer(self, answer):
        """Mã hóa câu trả lời"""
        if self.use_transformer_tokenizer:
            # Đoạn code này không được gọi vì chúng ta không sử dụng transformer tokenizer
            pass
        else:
            # Sử dụng NLTK tokenize đơn giản
            tokens = self.tokenize_text(answer)
            return tokens
    
    def process_questions_file(self, questions_file, output_file=None):
        """Xử lý file câu hỏi và trả lời"""
        processed_data = []
        
        # Đọc file câu hỏi
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                # Xử lý câu hỏi và câu trả lời
                processed_question = self.process_question(item['question'])
                processed_answer = self.process_answer(item['answer'])
                
                processed_item = {
                    'image_id': item['image_id'],
                    'question': processed_question,
                    'answer': processed_answer,
                    'original_question': item['question'],
                    'original_answer': item['answer']
                }
                
                processed_data.append(processed_item)
        
        # Lưu dữ liệu đã xử lý nếu cần
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')
        
        return processed_data
    
    def process_dataset(self, questions_file, split="train"):
        """Tiền xử lý tất cả câu hỏi và câu trả lời trong dataset"""
        processed_dir = self.processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        output_file = processed_dir / f"{split}_processed.jsonl"
        return self.process_questions_file(questions_file, output_file)