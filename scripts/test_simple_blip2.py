#!/usr/bin/env python
import os
import sys
import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.simple_model import SimpleBlip2VQA

def main():
    parser = argparse.ArgumentParser(description='Test Simple BLIP-2 VQA')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image (optional)')
    parser.add_argument('--question', type=str, default="What can you see in this medical image?",
                        help='Question to ask (optional)')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('simple_blip2_test', config['logging']['save_dir'], level='INFO')
    logger.info("Starting Simple BLIP-2 VQA test")
    
    # Kiểm tra CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Khởi tạo mô hình
        model = SimpleBlip2VQA(config)
        
        # Tìm hình ảnh test nếu không được cung cấp
        if args.image is None:
            test_image_dir = config['data']['test_images']
            test_images = list(Path(test_image_dir).glob("*.*"))
            if test_images:
                args.image = str(test_images[0])
                logger.info(f"Using test image: {args.image}")
            else:
                logger.error("No test images found")
                return
        
        # Dự đoán
        logger.info(f"Question: {args.question}")
        answer = model.predict(args.image, args.question)
        logger.info(f"Answer: {answer}")
        
        # Hiển thị kết quả
        plt.figure(figsize=(10, 8))
        img = Image.open(args.image).convert("RGB")
        plt.imshow(img)
        plt.title(f"Q: {args.question}\nA: {answer}")
        plt.axis("off")
        
        # Lưu hình ảnh
        output_dir = Path("data/tests")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "simple_blip2_test.png"
        plt.savefig(output_path)
        logger.info(f"Test visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    logger.info("Simple BLIP-2 VQA test completed")

if __name__ == "__main__":
    main()
