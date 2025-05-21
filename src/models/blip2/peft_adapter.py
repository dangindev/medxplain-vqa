import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class BLIP2LoRAVQA(nn.Module):
    """BLIP-2 với LoRA adapters cho Visual Question Answering"""
    
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lấy cấu hình mô hình
        self.model_name = config['model']['blip2']['pretrained_model_name']
        self.cache_dir = config['model']['blip2']['cache_dir']
        
        # Đảm bảo thư mục cache tồn tại
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initializing BLIP-2 model with LoRA from {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Tải processor
        try:
            self.processor = Blip2Processor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            logger.info("Loaded BLIP-2 processor successfully")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise
        
        # Cấu hình LoRA
        self.lora_config = LoraConfig(
            r=16,  # Rank của low-rank matrices
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Tải model
        try:
            base_model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Chuẩn bị mô hình cho kbit training nếu cần
            if config.get('training', {}).get('use_8bit', False):
                base_model = prepare_model_for_kbit_training(base_model)
            
            # Áp dụng LoRA
            self.model = get_peft_model(base_model, self.lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("Loaded BLIP-2 model with LoRA successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _process_inputs(self, images, questions):
        """Xử lý đầu vào cho mô hình"""
        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True)
        
        # Chuyển inputs sang device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """Forward pass qua mô hình BLIP-2 với LoRA"""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate_answer(self, image, question):
        """Generate câu trả lời từ hình ảnh và câu hỏi"""
        # Xử lý đầu vào
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding=True)
        
        # Chuyển inputs sang device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Generate câu trả lời
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=30,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode câu trả lời
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def save_adapter(self, save_dir):
        """Lưu adapters của mô hình"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Lưu LoRA adapters
        self.model.save_pretrained(save_path)
        
        logger.info(f"Saved LoRA adapters to {save_dir}")
    
    def load_adapter(self, load_dir):
        """Tải adapters đã lưu"""
        load_path = Path(load_dir)
        
        # Tải LoRA adapters
        self.model = PeftModel.from_pretrained(
            self.model,
            load_path,
            is_trainable=True
        )
        
        logger.info(f"Loaded LoRA adapters from {load_dir}")
