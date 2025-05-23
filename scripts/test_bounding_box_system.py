#!/usr/bin/env python
"""
Simple Bounding Box System Test
Using existing GradCAM + new BoundingBoxExtractor
Optimized for test_0001.jpg
"""

import os
import sys
import torch
import argparse
import json
from PIL import Image
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.explainability.enhanced_grad_cam import EnhancedGradCAM

def load_blip_model(config, model_path, logger):
    """Load BLIP model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading BLIP model from: {model_path}")
    
    try:
        model = BLIP2VQA(config, train_mode=False)
        
        if os.path.isdir(model_path):
            model.model = type(model.model).from_pretrained(model_path)
            model.model.to(device)
        else:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.model.load_state_dict(checkpoint)
        
        model.model.eval()
        
        # Add processor for GradCAM compatibility
        if not hasattr(model.model, 'processor'):
            model.model.processor = model.processor
        
        logger.info("BLIP model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def test_image_analysis(blip_model, image_path, question, output_dir, logger):
    """Test complete image analysis"""
    logger.info(f"Testing image: {image_path}")
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Image loaded: {image.size}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None
    
    # Test BLIP prediction
    try:
        blip_answer = blip_model.predict(image, question)
        logger.info(f"BLIP answer: '{blip_answer}'")
    except Exception as e:
        logger.error(f"BLIP prediction failed: {e}")
        blip_answer = "prediction_failed"
    
    # Initialize EnhancedGradCAM
    bbox_config = {
        'attention_threshold': 0.25,  # Lower for test_0001.jpg
        'min_region_size': 6,
        'max_regions': 5,
        'box_expansion': 0.12
    }
    
    try:
        enhanced_gradcam = EnhancedGradCAM(
            blip_model.model,
            bbox_config=bbox_config
        )
        logger.info("EnhancedGradCAM initialized")
    except Exception as e:
        logger.error(f"Failed to initialize EnhancedGradCAM: {e}")
        return None
    
    # Run analysis
    analysis_result = enhanced_gradcam.analyze_image_with_question(
        image, question, save_dir=output_dir
    )
    
    summary = enhanced_gradcam.get_summary(analysis_result)
    
    logger.info(f"Analysis completed: {summary}")
    return analysis_result, summary

def main():
    parser = argparse.ArgumentParser(description='Bounding Box System Test')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model-path', type=str, 
                       default='checkpoints/blip/checkpoints/best_hf_model')
    parser.add_argument('--test-image', type=str, 
                       default='data/images/test/test_0001.jpg')
    parser.add_argument('--question', type=str, 
                       default='What does this image show?')
    parser.add_argument('--output-dir', type=str, 
                       default='data/bbox_test_results')
    
    args = parser.parse_args()
    
    # Setup
    config = Config(args.config)
    logger = setup_logger('bbox_test', config['logging']['save_dir'])
    
    logger.info("🚀 BOUNDING BOX SYSTEM TEST")
    logger.info(f"Image: {args.test_image}")
    logger.info(f"Question: {args.question}")
    
    # Check if test image exists
    if not os.path.exists(args.test_image):
        logger.error(f"Test image not found: {args.test_image}")
        return
    
    # Load model
    blip_model = load_blip_model(config, args.model_path, logger)
    if not blip_model:
        logger.error("Failed to load BLIP model")
        return
    
    # Run test
    result = test_image_analysis(
        blip_model, args.test_image, args.question, args.output_dir, logger
    )
    
    if result:
        analysis_result, summary = result
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        results_data = {
            'test_image': args.test_image,
            'question': args.question,
            'analysis_result': {
                'success': analysis_result['success'],
                'regions_found': len(analysis_result.get('regions', [])),
                'error': analysis_result.get('error')
            },
            'summary': summary
        }
        
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Print summary
        logger.info("🎉 TEST COMPLETED")
        logger.info(f"✅ Success: {analysis_result['success']}")
        logger.info(f"📊 Regions found: {len(analysis_result.get('regions', []))}")
        logger.info(f"📁 Results saved to: {args.output_dir}")
        
        if analysis_result.get('visualization_path'):
            logger.info(f"🖼️ Visualization: {analysis_result['visualization_path']}")
    else:
        logger.error("❌ Test failed")

if __name__ == "__main__":
    main()
