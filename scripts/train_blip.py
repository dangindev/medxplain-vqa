#!/usr/bin/env python
import os
import sys
import argparse
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.trainer import BLIPTrainer

def plot_training_history(trainer, output_dir):
    """Vẽ đồ thị quá trình huấn luyện"""
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Vẽ loss
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    # Vẽ accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_accuracies, label='Train Accuracy')
    plt.plot(trainer.val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_history.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train BLIP model on PathVQA')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to the config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training')
    parser.add_argument('--output-dir', type=str, default='data/training_results',
                      help='Directory to save training results')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('train_blip', config['logging']['save_dir'], level='INFO')
    logger.info("Starting BLIP training")
    
    # Kiểm tra CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training on CPU may be slow.")
    
    # Khởi tạo trainer
    logger.info("Initializing trainer...")
    trainer = BLIPTrainer(config)
    
    # Tải checkpoint nếu có
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Huấn luyện mô hình
    logger.info("Starting training...")
    trainer.train()
    
    # Vẽ đồ thị quá trình huấn luyện
    logger.info("Plotting training history...")
    plot_training_history(trainer, args.output_dir)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()