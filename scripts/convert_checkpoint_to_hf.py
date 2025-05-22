#!/usr/bin/env python
import os
import sys
import torch
import argparse
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

def convert_checkpoint_to_hf(checkpoint_path, output_dir, config, logger):
    """Convert PyTorch checkpoint to HuggingFace format"""
    try:
        # Tải checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Khởi tạo mô hình
        model = BLIP2VQA(config, train_mode=False)
        
        # Load state dict
        model.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state dict loaded successfully")
        
        # Tạo thư mục đầu ra
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu dưới dạng HuggingFace
        model.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to HuggingFace format')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/blip/checkpoints/best_model.pth', 
                      help='Path to PyTorch checkpoint')
    parser.add_argument('--output-dir', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Output directory for HuggingFace format')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('convert_checkpoint', config['logging']['save_dir'], level='INFO')
    logger.info("Starting checkpoint conversion")
    
    # Convert checkpoint
    success = convert_checkpoint_to_hf(args.checkpoint, args.output_dir, config, logger)
    
    if success:
        logger.info("Checkpoint conversion completed successfully")
    else:
        logger.error("Checkpoint conversion failed")

if __name__ == "__main__":
    main()
