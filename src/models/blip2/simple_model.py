import os
import torch
import torch.nn as nn
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class SimpleBlip2VQA:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tải mô hình từ transformers
        model_name = "Salesforce/blip2-flan-t5-xl"  # Mô hình với T5
        
        logger.info(f"Loading simplified BLIP-2 model: {model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def predict(self, image_path, question):
        """
        Dự đoán câu trả lời cho một hình ảnh và câu hỏi
        
        Args:
            image_path: Đường dẫn đến hình ảnh
            question: Câu hỏi
            
        Returns:
            answer: Câu trả lời
        """
        try:
            # Tải hình ảnh
            image = Image.open(image_path).convert("RGB")
            
            # Chuẩn bị đầu vào
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
            
            # Tạo câu trả lời
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                
            # Giải mã câu trả lời
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return answer
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "Error generating answer"
