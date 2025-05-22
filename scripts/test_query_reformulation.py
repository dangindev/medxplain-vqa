#!/usr/bin/env python
import os
import sys
import argparse
from PIL import Image
from pathlib import Path
import json

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
    parser = argparse.ArgumentParser(description='Test Query Reformulation System')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--question', type=str, required=True, help='Question to reformulate')
    parser.add_argument('--output-dir', type=str, default='data/query_reformulation_test', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('test_query_reformulation', config['logging']['save_dir'], level='INFO')
    logger.info("Testing Query Reformulation System")
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Load BLIP model
        blip_model = BLIP2VQA(config, train_mode=False)
        model_path = 'checkpoints/blip/checkpoints/best_hf_model'
        if os.path.isdir(model_path):
            blip_model.model = type(blip_model.model).from_pretrained(model_path)
            blip_model.model.to(blip_model.device)
        blip_model.model.eval()
        
        # Initialize Gemini
        gemini = GeminiIntegration(config)
        
        # Initialize visual context extractor
        visual_extractor = VisualContextExtractor(blip_model, config)
        
        # Initialize query reformulator
        query_reformulator = QueryReformulator(gemini, visual_extractor, config)
        
        # Initialize question enhancer
        question_enhancer = QuestionEnhancer(query_reformulator, config)
        
        logger.info("All components initialized successfully")
        
        # Load and process image
        logger.info(f"Loading image: {args.image}")
        image = Image.open(args.image).convert('RGB')
        
        # Test visual context extraction
        logger.info("Testing visual context extraction...")
        visual_context = visual_extractor.extract_complete_context(image, args.question)
        logger.info(f"Visual context: {visual_context['visual_description']}")
        logger.info(f"Anatomical context: {visual_context['anatomical_context']}")
        
        # Test query reformulation
        logger.info("Testing query reformulation...")
        reformulation_result = query_reformulator.reformulate_question(image, args.question)
        
        logger.info("=== REFORMULATION RESULTS ===")
        logger.info(f"Original: {reformulation_result['original_question']}")
        logger.info(f"Reformulated: {reformulation_result['reformulated_question']}")
        logger.info(f"Question Type: {reformulation_result['question_type']}")
        logger.info(f"Quality Score: {reformulation_result['reformulation_quality']['score']:.3f}")
        
        # Test question enhancement
        logger.info("Testing question enhancement...")
        enhanced_result = question_enhancer.enhance_single_question(image, args.question, save_intermediate=True)
        
        logger.info("=== ENHANCEMENT RESULTS ===")
        logger.info(f"Enhancement Success: {enhanced_result['enhancement_success']}")
        logger.info(f"Overall Quality Score: {enhanced_result['quality_metrics']['overall_score']:.3f}")
        logger.info(f"Length Improvement: {enhanced_result['quality_metrics']['length_improvement']:.2f}x")
        
        if enhanced_result['validation_issues']:
            logger.info(f"Validation Issues: {enhanced_result['validation_issues']}")
        if enhanced_result['recommendations']:
            logger.info(f"Recommendations: {enhanced_result['recommendations']}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = {
            'test_metadata': {
                'image_path': args.image,
                'original_question': args.question
            },
            'visual_context': {
                'description': visual_context['visual_description'],
                'anatomical_context': visual_context['anatomical_context']
            },
            'reformulation_result': {
                'original_question': reformulation_result['original_question'],
                'reformulated_question': reformulation_result['reformulated_question'],
                'question_type': reformulation_result['question_type'],
                'quality_score': reformulation_result['reformulation_quality']['score']
            },
            'enhancement_result': enhanced_result
        }
        
        output_file = os.path.join(args.output_dir, 'query_reformulation_test_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to {output_file}")
        logger.info("Query Reformulation System test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
