import os
import torch
import torch.nn as nn
import logging
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

logger = logging.getLogger(__name__)

class BLIP2VQA(nn.Module):
    """BLIP model cho Visual Question Answering"""
    
    def __init__(self, config, train_mode=False):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_mode = train_mode
        self.max_length = config['model']['blip2']['max_answer_length']
        
        # Tải model và processor
        model_name = config['model']['blip2']['pretrained_model_name']
        cache_dir = config['model']['blip2']['cache_dir']
        
        logger.info(f"Loading BLIP model: {model_name}")
        
        # Tải processor và model
        try:
            # Tải BLIP processor
            self.processor = BlipProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Tải BLIP model
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Đưa mô hình lên thiết bị phù hợp
            self.model.to(self.device)
            
            # Cấu hình đóng băng (freeze) các thành phần
            self._configure_freezing()
            
            logger.info(f"BLIP model loaded successfully on {self.device}")
            
            # Thông tin mô hình
            self.num_parameters = self._count_parameters()
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            raise
    
    def _configure_freezing(self):
        """Cấu hình việc đóng băng các thành phần của mô hình"""
        # Đóng băng vision encoder nếu cần
        if self.config['model']['blip2']['freeze_vision_encoder']:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision encoder is frozen")
    
    def _count_parameters(self):
        """Đếm tổng số tham số của mô hình"""
        return sum(p.numel() for p in self.model.parameters())
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """Forward pass của mô hình BLIP"""
        # Đưa dữ liệu lên device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        
        # Chuyển đổi inputs để phù hợp với BLIP
        if labels is not None and self.train_mode:
            labels = labels.to(self.device)
            
            # Gọi model với labels
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True
            )
            
            return outputs
        else:
            # Gọi model không có labels
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True
            )
            
            return outputs
    
    def generate_answers(self, pixel_values, input_ids, attention_mask=None, 
                        max_length=None, num_beams=5):
        """Sinh câu trả lời từ mô hình BLIP"""
        # Đưa dữ liệu lên thiết bị
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Sử dụng max_length được truyền vào hoặc giá trị mặc định
        if max_length is None:
            max_length = self.max_length
        
        # Sinh câu trả lời
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    temperature=1.0
                )
                
                # Giải mã câu trả lời - xử lý mỗi câu trả lời riêng biệt
                answers = []
                for ids in generated_ids:
                    answer = self.processor.decode(ids, skip_special_tokens=True)
                    answers.append(answer)
                
                return answers
            except Exception as e:
                logger.error(f"Error generating answers: {e}")
                # Trả về câu trả lời rỗng nếu có lỗi
                return [""] * pixel_values.size(0)
    
    def predict(self, image, question, max_length=None, return_tensors=False):
        """
        Dự đoán câu trả lời cho một cặp hình ảnh và câu hỏi
        
        Args:
            image: PIL Image
            question: Câu hỏi string
            max_length: Độ dài tối đa của câu trả lời
            return_tensors: Có trả về tensor inputs không (cho Grad-CAM)
            
        Returns:
            answer: Câu trả lời được dự đoán
            inputs (optional): Tensor inputs nếu return_tensors=True
        """
        # Xử lý đầu vào và đưa vào đúng thiết bị
        inputs = self.processor(image, question, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)  # Đảm bảo chuyển vào đúng device
        
        # Sinh câu trả lời
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(**inputs, max_length=max_length or self.max_length)
                
                # Giải mã câu trả lời
                answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                if return_tensors:
                    return answer, inputs
                else:
                    return answer
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                if return_tensors:
                    return "", inputs
                else:
                    return ""

    def save_pretrained(self, output_dir):
        """Lưu model và processor"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu model
        self.model.save_pretrained(output_dir)
        
        # Lưu processor
        self.processor.save_pretrained(output_dir)
        
        logger.info(f"Model and processor saved to {output_dir}")
        
    def to(self, device):
        """Chuyển mô hình sang thiết bị cụ thể"""
        self.device = device
        self.model.to(device)
        return self
