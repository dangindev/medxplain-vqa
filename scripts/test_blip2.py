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
from src.models.blip2.model import BLIP2VQA

def test_blip2_load(config, logger):
    """Kiểm tra việc tải mô hình BLIP-2"""
    try:
        logger.info("Testing BLIP-2 VQA model loading...")
        start_time = time.time()
        
        # Cố gắng tải mô hình ở chế độ đánh giá (inference)
        model = BLIP2VQA(config, train_mode=False)
        
        end_time = time.time()
        
        logger.info(f"BLIP-2 model loaded successfully in {end_time - start_time:.2f} seconds")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load BLIP-2 model: {e}")
        logger.error(traceback.format_exc())
        return None

def test_inference(model, image_path, question, logger):
    """Kiểm tra inference với mô hình BLIP-2"""
    try:
        logger.info(f"Testing inference with image: {image_path}")
        logger.info(f"Question: {question}")
        
        # Tải hình ảnh từ đường dẫn
        image = Image.open(image_path).convert('RGB')
        
        # Dự đoán câu trả lời
        start_time = time.time()
        answer = model.predict(image, question)
        end_time = time.time()
        
        logger.info(f"Answer: {answer}")
        logger.info(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
        
        # Hiển thị kết quả
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f"Q: {question}\nA: {answer}")
        plt.axis("off")
        
        # Lưu hình ảnh
        output_dir = Path("data/tests")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "blip2_inference_test.png"
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Test BLIP-2 VQA model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image (optional)')
    parser.add_argument('--question', type=str, default=None,
                        help='Question to ask (optional)')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('blip2_test', config['logging']['save_dir'], level='INFO')
    logger.info("Starting BLIP-2 VQA model test")
    
    # Kiểm tra CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Tải mô hình
    model = test_blip2_load(config, logger)
    
    if model is not None:
        # Kiểm tra inference nếu có đường dẫn hình ảnh và câu hỏi
        if args.image and args.question:
            test_inference(model, args.image, args.question, logger)
        elif args.image is None and args.question is None:
            # Lấy một hình ảnh test từ dữ liệu
            try:
                test_image_dir = config['data']['test_images']
                test_image_paths = list(Path(test_image_dir).glob("*.*"))
                
                if test_image_paths:
                    test_image_path = test_image_paths[0]
                    test_question = "What abnormality can be seen in this pathology image?"
                    test_inference(model, test_image_path, test_question, logger)
                else:
                    logger.error(f"No test images found in {test_image_dir}")
            except Exception as e:
                logger.error(f"Could not find test image: {e}")
    
    logger.info("BLIP-2 VQA model test completed")

if __name__ == "__main__":
    main()