# scripts/test_grad_cam.py
#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.explainability.blip_cam import BLIPCAM

def main():
    parser = argparse.ArgumentParser(description='Test Grad-CAM on BLIP model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--question', type=str, required=True, help='Question to ask')
    parser.add_argument('--output-dir', type=str, default='data/grad_cam_results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('test_grad_cam', config['logging']['save_dir'], level='INFO')
    logger.info("Starting Grad-CAM test")
    
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
    
    # Khởi tạo BLIP-CAM
    logger.info("Initializing BLIP-CAM")
    blip_cam = BLIPCAM(blip_model)
    
    # Tải hình ảnh
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # Generate CAM
    logger.info(f"Generating CAM for question: {args.question}")
    cam, outputs = blip_cam.generate_cam(image, args.question)
    
    # Create overlay
    logger.info("Creating overlay")
    overlay = blip_cam.overlay_cam(image, cam)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize and save results
    save_path = os.path.join(args.output_dir, "grad_cam_visualization.png")
    blip_cam.visualize(image, cam, overlay, save_path)
    
    # Get prediction
    logger.info("Getting prediction")
    prediction = blip_model.predict(image, args.question)
    logger.info(f"BLIP prediction: {prediction}")
    
    # Clean up
    blip_cam.remove_hooks()
    
    logger.info("Grad-CAM test completed")

if __name__ == "__main__":
    main()