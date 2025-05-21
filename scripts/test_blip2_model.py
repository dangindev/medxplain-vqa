#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
import traceback

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger

def test_simple_model(config, logger):
    try:
        logger.info("Testing basic model load")
        
        # Tải ViT model đơn giản từ transformers 
        from transformers import ViTModel, AutoImageProcessor
        
        logger.info("Importing ViT model")
        model_name = "google/vit-base-patch16-224"
        
        logger.info(f"Loading image processor for {model_name}")
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        logger.info(f"Loading model {model_name}")
        model = ViTModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("Model moved to CUDA")
        
        num_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {num_parameters:,} parameters")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Simple Vision Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('model_test', config['logging']['save_dir'], level='INFO')
    logger.info("Starting basic model test")
    
    # Kiểm tra CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Test mô hình đơn giản
    success = test_simple_model(config, logger)
    
    if success:
        logger.info("Basic model test completed successfully")
    else:
        logger.error("Basic model test failed")

if __name__ == "__main__":
    main()