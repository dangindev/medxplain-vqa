#!/usr/bin/env python
import os
import sys
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.trainer import BLIPTrainer

def main():
    parser = argparse.ArgumentParser(description='Continue training BLIP model with improved prompting')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_model.pth', 
                      help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='data/improved_training', 
                      help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=5, help='Number of additional epochs')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Điều chỉnh một số tham số
    config.config['training']['learning_rate'] = 2e-5  # Giảm learning rate
    config.config['training']['num_epochs'] = args.epochs
    
    # Setup logger
    logger = setup_logger('improved_training', config['logging']['save_dir'], level='INFO')
    logger.info("Starting improved training")
    
    # Khởi tạo trainer
    trainer = BLIPTrainer(config)
    
    # Tải checkpoint nếu đã tồn tại
    if os.path.exists(args.model_path):
        logger.info(f"Loading checkpoint from {args.model_path}")
        trainer.load_checkpoint(args.model_path)
    
    # Training
    trainer.train()
    
    logger.info("Improved training completed")

if __name__ == "__main__":
    main()
