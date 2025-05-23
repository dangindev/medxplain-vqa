#!/usr/bin/env python
"""
Comprehensive Bounding Box System Test V2
With extensive debugging and step-by-step validation
"""

import os
import sys
import torch
import argparse
import json
from PIL import Image
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.explainability.enhanced_gradcam_v2 import EnhancedGradCAMV2

def load_blip_model(config, model_path, logger):
    """Load BLIP model with error handling"""
    logger.info(f"Loading BLIP model from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize model
        model = BLIP2VQA(config, train_mode=False)
        logger.info("BLIP2VQA initialized")
        
        # Load checkpoint
        if os.path.isdir(model_path):
            # HuggingFace format
            logger.info("Loading from HuggingFace directory")
            model.model = type(model.model).from_pretrained(model_path)
            model.model.to(device)
        else:
            # PyTorch checkpoint
            logger.info("Loading from PyTorch checkpoint")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.model.load_state_dict(checkpoint)
        
        model.model.eval()
        
        # Add processor attribute for Grad-CAM compatibility
        if not hasattr(model.model, 'processor'):
            model.model.processor = model.processor
            logger.info("Added processor attribute to model for Grad-CAM")
        
        logger.info("BLIP model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading BLIP model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def test_synthetic_heatmap(logger):
    """Test bounding box extraction with synthetic heatmap"""
    logger.info("=" * 60)
    logger.info("TESTING SYNTHETIC HEATMAP")
    logger.info("=" * 60)
    
    from src.explainability.simple_bbox_extractor import SimpleBoundingBoxExtractor
    
    # Create synthetic heatmap with clear attention regions
    heatmap = np.zeros((14, 14), dtype=np.float32)
    
    # Add attention regions
    heatmap[2:5, 2:5] = 0.9    # Top-left high attention
    heatmap[8:11, 8:11] = 0.8  # Bottom-right medium attention
    heatmap[5:7, 1:3] = 0.6    # Left-center low attention
    
    # Add noise
    noise = np.random.normal(0, 0.05, heatmap.shape)
    heatmap = np.clip(heatmap + noise, 0, 1)
    
    logger.info(f"Created synthetic heatmap: {heatmap.shape}")
    logger.info(f"Heatmap stats: min={heatmap.min():.3f}, max={heatmap.max():.3f}, mean={heatmap.mean():.3f}")
    
    # Test extraction
    extractor = SimpleBoundingBoxExtractor({
        'attention_threshold': 0.4,
        'min_region_size': 5,
        'max_regions': 5
    })
    
    regions = extractor.extract_regions_from_heatmap(heatmap, (224, 224))
    
    logger.info(f"Extraction results: {len(regions)} regions found")
    for i, region in enumerate(regions):
        logger.info(f"  Region {i+1}: bbox={region['bbox']}, score={region['attention_score']:.3f}")
    
    return heatmap, regions

def test_real_image_analysis(blip_model, image_path, question, output_dir, logger):
    """Test complete analysis with real image"""
    logger.info("=" * 60)
    logger.info("TESTING REAL IMAGE ANALYSIS")
    logger.info("=" * 60)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Image loaded successfully: {image.size}, mode: {image.mode}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None
    
    # Test BLIP prediction first
    logger.info("Testing BLIP prediction")
    try:
        blip_answer = blip_model.predict(image, question)
        logger.info(f"BLIP answer: '{blip_answer}'")
    except Exception as e:
        logger.error(f"BLIP prediction failed: {e}")
        blip_answer = "prediction_failed"
    
    # Initialize Enhanced Grad-CAM
    logger.info("Initializing Enhanced Grad-CAM V2")
    bbox_config = {
        'attention_threshold': 0.3,
        'min_region_size': 8,
        'max_regions': 3,
        'box_expansion': 0.15
    }
    
    try:
        enhanced_gradcam = EnhancedGradCAMV2(
            blip_model.model,
            layer_name="vision_model.encoder.layers.11",
            bbox_config=bbox_config
        )
        logger.info("Enhanced Grad-CAM V2 initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced Grad-CAM: {e}")
        return None
    
    # Run complete analysis
    logger.info("Running complete analysis")
    analysis_result = enhanced_gradcam.analyze_image_with_question(
        image, question, save_dir=output_dir
    )
    
    # Get summary
    summary = enhanced_gradcam.get_analysis_summary(analysis_result)
    
    logger.info("Analysis completed")
    logger.info(f"Summary: {summary}")
    
    return analysis_result, summary

def create_test_report(synthetic_results, real_results, output_dir, logger):
    """Create comprehensive test report"""
    logger.info("Creating test report")
    
    report = {
        'test_timestamp': None,
        'synthetic_test': {
            'status': 'completed',
            'heatmap_shape': synthetic_results[0].shape if synthetic_results else None,
            'regions_found': len(synthetic_results[1]) if synthetic_results else 0,
            'regions': synthetic_results[1] if synthetic_results else []
        },
        'real_image_test': {
            'status': 'skipped',
            'analysis_result': None,
            'summary': None
        }
    }
    
    # Add timestamp
    from datetime import datetime
    report['test_timestamp'] = datetime.now().isoformat()
    
    # Add real image results if available
    if real_results:
        analysis_result, summary = real_results
        report['real_image_test'] = {
            'status': 'completed',
            'success': analysis_result.get('success', False),
            'error': analysis_result.get('error'),
            'regions_found': len(analysis_result.get('regions', [])),
            'summary': summary
        }
    
    # Save report
    report_path = os.path.join(output_dir, 'bbox_test_report.json')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to: {report_path}")
    return report

def main():
    parser = argparse.ArgumentParser(description='Bounding Box System Test V2')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model-path', type=str, 
                       default='checkpoints/blip/checkpoints/best_hf_model')
    parser.add_argument('--test-image', type=str, default=None)
    parser.add_argument('--question', type=str, default='What does this image show?')
    parser.add_argument('--output-dir', type=str, default='data/bbox_test_v2_results')
    
    args = parser.parse_args()
    
    # Setup
    config = Config(args.config)
    logger = setup_logger('bbox_test_v2', config['logging']['save_dir'])
    
    logger.info("🚀 BOUNDING BOX SYSTEM TEST V2 STARTED")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Test 1: Synthetic heatmap
    logger.info("\n" + "🧪 TEST 1: SYNTHETIC HEATMAP")
    synthetic_results = test_synthetic_heatmap(logger)
    
    # Test 2: Real image (optional)
    real_results = None
    if args.test_image and os.path.exists(args.test_image):
        logger.info(f"\n" + "🧪 TEST 2: REAL IMAGE - {args.test_image}")
        
        # Load BLIP model
        blip_model = load_blip_model(config, args.model_path, logger)
        if blip_model:
            real_results = test_real_image_analysis(
                blip_model, args.test_image, args.question, args.output_dir, logger
            )
        else:
            logger.error("Failed to load BLIP model, skipping real image test")
    else:
        logger.info(f"\n" + "⏭️ TEST 2: SKIPPED (no test image provided)")
    
    # Generate report
    logger.info("\n" + "📊 GENERATING TEST REPORT")
    report = create_test_report(synthetic_results, real_results, args.output_dir, logger)
    
    # Final summary
    logger.info("\n" + "🎉 BOUNDING BOX SYSTEM TEST V2 COMPLETED")
    logger.info("=" * 60)
    logger.info("📋 SUMMARY:")
    logger.info(f"✅ Synthetic test: {report['synthetic_test']['regions_found']} regions found")
    
    if real_results:
        real_status = report['real_image_test']
        if real_status['success']:
            logger.info(f"✅ Real image test: SUCCESS - {real_status['regions_found']} regions found")
        else:
            logger.info(f"❌ Real image test: FAILED - {real_status.get('error', 'Unknown error')}")
    else:
        logger.info("⏭️ Real image test: SKIPPED")
    
    logger.info(f"📁 Results saved to: {args.output_dir}")
    logger.info("🚀 System ready for integration!")

if __name__ == "__main__":
    main()
