#!/usr/bin/env python
import os
import sys
import argparse
import json
import random
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration
from src.explainability.reasoning.visual_context_extractor import VisualContextExtractor
from src.explainability.reasoning.query_reformulator import QueryReformulator
from src.explainability.reasoning.question_enhancer import QuestionEnhancer

def main():
    parser = argparse.ArgumentParser(description='Test Query Reformulation with Batch Processing')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train', 'val'], help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--output-dir', type=str, default='data/query_reformulation_batch_test', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('test_query_reformulation_batch', config['logging']['save_dir'], level='INFO')
    logger.info("Testing Query Reformulation System with Batch Processing")
    
    try:
        # Get correct dataset path from config
        dataset_path = config['data'][f'{args.split}_questions']
        images_dir = config['data'][f'{args.split}_images']
        
        logger.info(f"Using dataset: {dataset_path}")
        logger.info(f"Using images dir: {images_dir}")
        
        # Verify paths exist
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Load BLIP model
        blip_model = BLIP2VQA(config, train_mode=False)
        model_path = 'checkpoints/blip/checkpoints/best_hf_model'
        if os.path.isdir(model_path):
            blip_model.model = type(blip_model.model).from_pretrained(model_path)
            blip_model.model.to(blip_model.device)
        blip_model.model.eval()
        
        # Initialize other components
        gemini = GeminiIntegration(config)
        visual_extractor = VisualContextExtractor(blip_model, config)
        query_reformulator = QueryReformulator(gemini, visual_extractor, config)
        question_enhancer = QuestionEnhancer(query_reformulator, config)
        
        logger.info("All components initialized successfully")
        
        # Test dataset enhancement
        logger.info(f"Testing dataset enhancement with {args.num_samples} samples from {args.split} split")
        summary = question_enhancer.enhance_dataset_questions(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            max_samples=args.num_samples
        )
        
        logger.info("=== BATCH PROCESSING SUMMARY ===")
        logger.info(f"Total Questions: {summary['total_questions']}")
        logger.info(f"Successfully Enhanced: {summary['successfully_enhanced']}")
        logger.info(f"Failed Processing: {summary['failed_processing']}")
        logger.info(f"Success Rate: {summary['success_rate']:.2%}")
        logger.info(f"Average Quality Score: {summary['average_quality_score']:.3f}")
        
        # Get enhancement statistics
        stats = question_enhancer.get_enhancement_statistics()
        logger.info("=== ENHANCEMENT STATISTICS ===")
        logger.info(f"Total Processed: {stats['total_processed']}")
        logger.info(f"Successful: {stats['successful_reformulations']}")
        logger.info(f"Failed: {stats['failed_reformulations']}")
        logger.info(f"Average Quality: {stats['average_quality_score']:.3f}")
        
        logger.info("Batch Query Reformulation test completed successfully!")
        
    except Exception as e:
        logger.error(f"Batch test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
