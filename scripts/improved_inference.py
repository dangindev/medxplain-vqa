#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

def improved_prompt(question):
    """Tạo prompt tốt hơn từ câu hỏi gốc"""
    # Thêm prefix để hướng dẫn mô hình trả lời chi tiết
    return f"Answer the following medical pathology question with a detailed explanation: {question}"

def main():
    parser = argparse.ArgumentParser(description='Test model with better prompting')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--question', type=str, required=True, help='Question to ask')
    parser.add_argument('--output-dir', type=str, default='data/improved_inference', 
                      help='Directory to save results')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('improved_inference', config['logging']['save_dir'], level='INFO')
    logger.info("Starting improved inference")
    
    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Tải mô hình
    logger.info(f"Loading model from {args.model_path}")
    model = BLIP2VQA(config, train_mode=False)
    model.device = device
    
    if os.path.isdir(args.model_path):
        model.model = type(model.model).from_pretrained(args.model_path)
        model.model.to(device)
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.model.load_state_dict(checkpoint)
    
    model.model.eval()
    logger.info("Model loaded successfully")
    
    # Tải và xử lý hình ảnh
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # Dự đoán câu trả lời với prompt cải tiến
    improved_question = improved_prompt(args.question)
    logger.info(f"Original question: {args.question}")
    logger.info(f"Improved question: {improved_question}")
    
    answer = model.predict(image, improved_question)
    logger.info(f"Predicted answer: {answer}")
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Hiển thị và lưu kết quả
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"Q: {args.question}\nA: {answer}")
    plt.axis('off')
    
    output_path = os.path.join(args.output_dir, "improved_prediction.png")
    plt.savefig(output_path)
    logger.info(f"Result saved to {output_path}")
    
    logger.info("Improved inference completed")

if __name__ == "__main__":
    main()
