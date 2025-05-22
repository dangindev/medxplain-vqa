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
from src.explainability.visualization import (
    visualize_gradcam,
    get_salient_regions,
    describe_salient_regions
)

def main():
    parser = argparse.ArgumentParser(description='Custom Explainable MedXplain-VQA')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--question', type=str, required=True, help='Question to answer')
    parser.add_argument('--output-dir', type=str, default='data/custom_explainable_results', help='Output directory')
    parser.add_argument('--target-layer', type=str, default="vision_model.encoder.layers.11", 
                      help='Target layer for Grad-CAM')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('custom_explainable_vqa', config['logging']['save_dir'], level='INFO')
    logger.info("Starting Custom Explainable MedXplain-VQA")
    
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
    logger.info(f"Initializing Grad-CAM with target layer: {args.target_layer}")
    grad_cam = GradCAM(blip_model.model, layer_name=args.target_layer)
    
    # Khởi tạo Gemini
    logger.info("Initializing Gemini")
    gemini = GeminiIntegration(config)
    
    # Tải hình ảnh
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # BLIP prediction
    logger.info(f"Question: {args.question}")
    blip_answer, inputs = blip_model.predict(image, args.question, return_tensors=True)
    logger.info(f"Initial BLIP answer: {blip_answer}")
    
    # Tạo Grad-CAM heatmap
    logger.info("Generating Grad-CAM heatmap...")
    heatmap = grad_cam(image, args.question, inputs, original_size=image.size)
    
    # Trích xuất và mô tả các vùng nổi bật
    if heatmap is not None:
        logger.info("Extracting salient regions...")
        regions = get_salient_regions(heatmap, threshold=0.5)
        region_descriptions = describe_salient_regions(regions, image.width, image.height)
        logger.info(f"Region descriptions: {region_descriptions}")
    else:
        logger.warning("Failed to generate Grad-CAM heatmap")
        regions = []
        region_descriptions = None
    
    # Tạo câu trả lời thống nhất với Gemini
    logger.info("Generating unified answer with Gemini...")
    unified_answer = gemini.generate_unified_answer(
        image, 
        args.question, 
        blip_answer, 
        heatmap=heatmap, 
        region_descriptions=region_descriptions
    )
    logger.info(f"Unified answer: {unified_answer}")
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo trực quan hóa tổng hợp
    logger.info("Creating visualization...")
    fig = plt.figure(figsize=(12, 12))
    
    # Grid layout: 2x2
    # Hình ảnh gốc
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis('off')
    
    # Grad-CAM heatmap
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    if heatmap is not None:
        ax2.imshow(heatmap, cmap='jet')
    else:
        ax2.text(0.5, 0.5, "Heatmap not available", ha='center', va='center')
    ax2.set_title("Attention Heatmap", fontsize=12)
    ax2.axis('off')
    
    # Text area với câu hỏi và câu trả lời
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    text_content = (
        f"Question: {args.question}\n\n"
        f"MedXplain-VQA answer: {unified_answer}"
    )
    ax3.text(0.01, 0.99, text_content, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', wrap=True)
    ax3.axis('off')
    
    # Lưu trực quan hóa tổng hợp
    plt.suptitle("MedXplain-VQA Analysis", fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(args.output_dir, "custom_explainable_result.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Visualization saved to {output_path}")
    
    # Gỡ bỏ hooks Grad-CAM
    grad_cam.remove_hooks()
    logger.info("Custom Explainable MedXplain-VQA completed")

if __name__ == "__main__":
    main()
