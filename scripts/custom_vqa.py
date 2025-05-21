#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration

def main():
    parser = argparse.ArgumentParser(description='MedXplain-VQA for custom image')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--question', type=str, required=True, help='Question to answer')
    parser.add_argument('--output-dir', type=str, default='data/custom_vqa_results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('custom_vqa', config['logging']['save_dir'], level='INFO')
    logger.info("Starting custom VQA analysis")
    
    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Tải mô hình BLIP
    logger.info(f"Loading BLIP model from {args.model_path}")
    blip_model = BLIP2VQA(config, train_mode=False)
    blip_model.device = device
    
    if os.path.isdir(args.model_path):
        blip_model.model = type(blip_model.model).from_pretrained(args.model_path)
        blip_model.model.to(device)
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            blip_model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            blip_model.model.load_state_dict(checkpoint)
    
    blip_model.model.eval()
    logger.info("BLIP model loaded successfully")
    
    # Khởi tạo Gemini
    logger.info("Initializing Gemini")
    gemini = GeminiIntegration(config)
    
    # Tải hình ảnh
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # BLIP prediction - bước trung gian
    logger.info(f"Question: {args.question}")
    blip_answer = blip_model.predict(image, args.question)
    logger.info(f"Initial BLIP answer: {blip_answer}")
    
    # Tạo câu trả lời thống nhất
    logger.info("Generating unified answer...")
    unified_answer = gemini.generate_unified_answer(image, args.question, blip_answer)
    logger.info(f"MedXplain-VQA answer: {unified_answer}")
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sử dụng cách tiếp cận khác để hiển thị - sử dụng subplot thay vì figtext
    fig = plt.figure(figsize=(10, 12))
    
    # Tạo hai phần: phần trên cho hình ảnh, phần dưới cho văn bản
    ax_image = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_text = plt.subplot2grid((3, 1), (2, 0))
    
    # Hiển thị hình ảnh
    ax_image.imshow(image)
    ax_image.set_title("MedXplain-VQA Analysis", fontsize=14)
    ax_image.axis('off')
    
    # Hiển thị văn bản trong phần dưới - sử dụng text box
    text_content = f"Question: {args.question}\n\nMedXplain-VQA answer: {unified_answer}"
    ax_text.text(0.01, 0.99, text_content, 
                transform=ax_text.transAxes,
                fontsize=11,
                verticalalignment='top',
                wrap=True)
    ax_text.axis('off')
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu kết quả
    output_path = os.path.join(args.output_dir, "custom_vqa_result.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
    logger.info(f"Result saved to {output_path}")
    
    logger.info("Custom VQA analysis completed")

if __name__ == "__main__":
    main()
