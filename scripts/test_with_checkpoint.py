#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration

def main():
    parser = argparse.ArgumentParser(description='Test with existing checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/blip/checkpoints/best_model.pth')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()
    
    config = Config(args.config)
    logger = setup_logger('test_checkpoint', config['logging']['save_dir'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Tải mô hình và checkpoint
    logger.info("Loading BLIP model...")
    blip_model = BLIP2VQA(config, train_mode=False)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    blip_model.model.load_state_dict(checkpoint['model_state_dict'])
    blip_model.model.eval()
    logger.info("Checkpoint loaded successfully")
    
    # Khởi tạo Gemini
    logger.info("Initializing Gemini...")
    gemini = GeminiIntegration(config)
    
    # Test
    image = Image.open(args.image).convert('RGB')
    logger.info(f"Question: {args.question}")
    
    blip_answer = blip_model.predict(image, args.question)
    logger.info(f"BLIP answer: {blip_answer}")
    
    unified_answer = gemini.generate_unified_answer(image, args.question, blip_answer)
    logger.info(f"MedXplain-VQA answer: {unified_answer}")
    
    print(f"\n{'='*50}")
    print(f"Question: {args.question}")
    print(f"Answer: {unified_answer}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
