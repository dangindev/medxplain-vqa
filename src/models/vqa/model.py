import os
import torch
import torch.nn as nn
import logging
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

logger = logging.getLogger(__name__)

class BLIPVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lấy cấu hình mô hình
        model_name = config['model']['vqa']['pretrained_model_name']
        cache_dir = config['model']['vqa']['cache_dir']
        
        logger.info(f"Loading BLIP-VQA model: {model_name}")
        
        # Tải processor và model
        try:
            self.processor = BlipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Đưa mô hình lên GPU nếu có
            self.model.to(self.device)
            
            logger.info(f"BLIP-VQA model loaded successfully on {self.device}")
            
            # Thông tin mô hình
            self.num_parameters = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Number of parameters: {self.num_parameters:,}")
        except Exception as e:
            logger.error(f"Error loading BLIP-VQA model: {e}")
            raise
    
    def predict(self, image, question, max_length=30):
        """
        Dự đoán câu trả lời cho một cặp hình ảnh và câu hỏi
        
        Args:
            image: PIL Image hoặc đường dẫn đến hình ảnh
            question: Câu hỏi string
            max_length: Độ dài tối đa của câu trả lời
            
        Returns:
            answer: Câu trả lời được dự đoán
        """
        # Xử lý đầu vào
        if isinstance(image, str):
            # Nếu là đường dẫn đến file hình ảnh
            image = Image.open(image).convert('RGB')
        elif isinstance(image, torch.Tensor):
            # Nếu là tensor, chuyển về PIL Image
            if image.dim() == 4:  # [B, C, H, W]
                image = image[0]  # Lấy mẫu đầu tiên nếu là batch
            
            # Chuyển tensor về numpy array và sau đó thành PIL Image
            image = image.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            
            # Denormalize nếu cần
            mean = self.config['preprocessing']['image']['normalize']['mean']
            std = self.config['preprocessing']['image']['normalize']['std']
            image = image * std + mean
            image = (image * 255).astype('uint8')
            image = Image.fromarray(image)
        
        # Xử lý đầu vào với processor
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Thực hiện inference
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)
        
        # Giải mã câu trả lời
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        
        return answer
    
    def forward_batch(self, images, questions, max_length=30):
        """
        Forward pass cho cả batch
        
        Args:
            images: List of PIL Images hoặc tensor [B, C, H, W]
            questions: List of strings
            max_length: Độ dài tối đa của câu trả lời
            
        Returns:
            answers: List các câu trả lời được dự đoán
        """
        batch_size = len(questions)
        answers = []
        
        for i in range(batch_size):
            if isinstance(images, torch.Tensor):
                image = images[i]
            else:
                image = images[i]
            
            question = questions[i]
            answer = self.predict(image, question, max_length)
            answers.append(answer)
        
        return answers
