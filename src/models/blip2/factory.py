import logging
import importlib
from pathlib import Path
import torch
import os

logger = logging.getLogger(__name__)

def create_model(config, device=None):
    """
    Factory function để tạo mô hình dựa trên cấu hình
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Kiểm tra xem có sử dụng PEFT/LoRA hay không
    use_lora = config.get('model', {}).get('blip2', {}).get('use_lora', False)
    
    try:
        if use_lora:
            # Kiểm tra xem peft đã được cài đặt chưa
            try:
                import peft
                from src.models.blip2.peft_adapter import BLIP2LoRAVQA
                logger.info("Creating BLIP-2 model with LoRA adapters...")
                return BLIP2LoRAVQA(config, device)
            except ImportError:
                logger.warning("PEFT library not found. Please install it with: pip install peft")
                logger.warning("Falling back to standard model...")
                from src.models.blip2.model import BLIP2VQA
                return BLIP2VQA(config, device)
        else:
            # Sử dụng mô hình chuẩn với adapters tự tạo
            from src.models.blip2.model import BLIP2VQA
            logger.info("Creating standard BLIP-2 model...")
            return BLIP2VQA(config, device)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def load_model_from_checkpoint(config, checkpoint_path, device=None):
    """
    Tải mô hình từ checkpoint
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    try:
        # Tạo mô hình mới
        model = create_model(config, device)
        
        # Tải checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Kiểm tra xem checkpoint có chứa state_dict không
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Nếu không, giả định checkpoint là state_dict
            model.load_state_dict(checkpoint)
        
        logger.info(f"Successfully loaded model from checkpoint: {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from checkpoint: {e}")
        raise
