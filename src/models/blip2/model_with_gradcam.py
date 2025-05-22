import os
import torch
import torch.nn as nn
import logging
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Import Grad-CAM
from src.explainability.grad_cam import GradCAM

logger = logging.getLogger(__name__)

class BLIP2VQAWithGradCAM(nn.Module):
    """BLIP model cho Visual Question Answering với Grad-CAM"""
    
    def __init__(self, config, train_mode=False):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_mode = train_mode
        self.max_length = config['model']['blip2']['max_answer_length']
        
        # Tải model và processor
        model_name = config['model']['blip2']['pretrained_model_name']
        cache_dir = config['model']['blip2']['cache_dir']
        
        logger.info(f"Loading BLIP model with Grad-CAM: {model_name}")
        
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
            
            # Khởi tạo GradCAM
            self.grad_cam = GradCAM(self, target_layer_name="vision_model.encoder.layers.11")
            
            logger.info(f"BLIP model with Grad-CAM loaded successfully on {self.device}")
            
            # Thông tin mô hình
            self.num_parameters = self._count_parameters()
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        except Exception as e:
            logger.error(f"Error loading BLIP model with Grad-CAM: {e}")
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
        """
        Forward pass của mô hình BLIP
        
        Args:
            input_ids: Input ids của câu hỏi [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pixel_values: Tensor hình ảnh [batch_size, 3, H, W]
            labels: Labels cho language modeling (optional)
            
        Returns:
            outputs: Kết quả từ mô hình BLIP
        """
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
    
    def predict_with_gradcam(self, image, question):
        """
        Dự đoán câu trả lời và tạo Grad-CAM cho một cặp hình ảnh và câu hỏi
        
        Args:
            image: PIL Image
            question: Câu hỏi string
            
        Returns:
            answer: Câu trả lời được dự đoán
            vis_image: Hình ảnh với heatmap
            heatmap: Heatmap
            cam: Class Activation Map gốc
        """
        # Dự đoán câu trả lời
        answer = self.predict(image, question)
        
        # Tạo Grad-CAM
        vis_image, heatmap, cam = self.grad_cam(image, question)
        
        return answer, vis_image, heatmap, cam
    
    def predict(self, image, question, max_length=None):
        """
        Dự đoán câu trả lời cho một cặp hình ảnh và câu hỏi
        
        Args:
            image: PIL Image
            question: Câu hỏi string
            max_length: Độ dài tối đa của câu trả lời
            
        Returns:
            answer: Câu trả lời được dự đoán
        """
        # Xử lý đầu vào và đưa vào đúng thiết bị
        inputs = self.processor(image, question, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        
        # Sinh câu trả lời
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(**inputs, max_length=max_length or self.max_length)
                
                # Giải mã câu trả lời
                answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                return answer
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                return ""

    def save_pretrained(self, output_dir):
        """
        Lưu model và processor
        
        Args:
            output_dir: Thư mục đầu ra
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu model
        self.model.save_pretrained(output_dir)
        
        # Lưu processor
        self.processor.save_pretrained(output_dir)
        
        logger.info(f"Model and processor saved to {output_dir}")
        
    def to(self, device):
        """
        Chuyển mô hình sang thiết bị cụ thể
        
        Args:
            device: Thiết bị đích
        """
        self.device = device
        self.model.to(device)
        return self