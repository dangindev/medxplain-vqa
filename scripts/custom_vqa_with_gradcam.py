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
from src.explainability.grad_cam import GradCAM

def main():
    parser = argparse.ArgumentParser(description='MedXplain-VQA with Grad-CAM for custom image')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--layer-name', type=str, default='vision_model.encoder.layers.11', 
                      help='Layer name for Grad-CAM')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--question', type=str, required=True, help='Question to answer')
    parser.add_argument('--output-dir', type=str, default='data/custom_gradcam_results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('custom_gradcam', config['logging']['save_dir'], level='INFO')
    logger.info("Starting custom VQA analysis with Grad-CAM")
    
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
    
    # Khởi tạo Grad-CAM
    logger.info(f"Initializing Grad-CAM for layer: {args.layer_name}")
    grad_cam = GradCAM(blip_model, args.layer_name)
    
    # Khởi tạo Gemini
    logger.info("Initializing Gemini")
    gemini = GeminiIntegration(config)
    
    # Tải hình ảnh
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image).convert('RGB')
    image_tensor = blip_model.processor(images=image, return_tensors="pt").pixel_values
    
    # BLIP prediction - bước trung gian
    logger.info(f"Question: {args.question}")
    blip_answer = blip_model.predict(image, args.question)
    logger.info(f"Initial BLIP answer: {blip_answer}")
    
    # Tạo Grad-CAM
    logger.info("Generating Grad-CAM heatmap...")
    cam, overlay = grad_cam(image_tensor, image)
    
    # Tạo câu trả lời thống nhất
    logger.info("Generating unified answer with Grad-CAM context...")
    unified_answer = gemini.generate_unified_answer(image, args.question, blip_answer, overlay)
    logger.info(f"MedXplain-VQA answer: {unified_answer}")
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Trực quan hóa
    fig = plt.figure(figsize=(15, 10))
    
    # Tạo 2x2 subplots
    ax_image = plt.subplot2grid((2, 2), (0, 0))
    ax_heatmap = plt.subplot2grid((2, 2), (0, 1))
    ax_overlay = plt.subplot2grid((2, 2), (1, 0))
    ax_text = plt.subplot2grid((2, 2), (1, 1))
    
    # Hiển thị hình ảnh gốc
    ax_image.imshow(image)
    ax_image.set_title("Original Image", fontsize=12)
    ax_image.axis('off')
    
    # Hiển thị heatmap
    ax_heatmap.imshow(cam, cmap='jet')
    ax_heatmap.set_title("Grad-CAM Heatmap", fontsize=12)
    ax_heatmap.axis('off')
    
    # Hiển thị overlay
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Heatmap Overlay", fontsize=12)
    ax_overlay.axis('off')
    
    # Hiển thị văn bản
    text_content = f"Question: {args.question}\n\nMedXplain-VQA answer: {unified_answer}"
    ax_text.text(0.01, 0.99, text_content, 
                transform=ax_text.transAxes,
                fontsize=10,
                verticalalignment='top',
                wrap=True)
    ax_text.axis('off')
    
    # Thêm siêu tiêu đề
    plt.suptitle("MedXplain-VQA Analysis with Grad-CAM", fontsize=14)
    
    # Điều chỉnh layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Để lại không gian cho siêu tiêu đề
    
    # Lưu kết quả
    output_path = os.path.join(args.output_dir, "custom_gradcam_result.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
    logger.info(f"Result saved to {output_path}")
    
    logger.info("Custom VQA analysis with Grad-CAM completed")

if __name__ == "__main__":
    main()