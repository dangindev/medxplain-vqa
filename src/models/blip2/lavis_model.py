import os
import torch
import torch.nn as nn
import logging
from PIL import Image
from lavis.models import load_model_and_preprocess

logger = logging.getLogger(__name__)

class LavisBlip2VQA:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading BLIP-2 model using LAVIS library")
        
        try:
            # Tải mô hình và bộ tiền xử lý
            self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name="blip2_t5", 
                model_type="pretrain_flant5xl", 
                is_eval=True,
                device=self.device
            )
            
            logger.info(f"LAVIS BLIP-2 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading LAVIS BLIP-2 model: {e}")
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
            # Tải và xử lý hình ảnh
            raw_image = Image.open(image_path).convert("RGB")
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
            
            # Dự đoán
            answer = self.model.generate({"image": image, "text_input": question})
            return answer[0]
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "Error generating answer"
