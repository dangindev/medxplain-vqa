#!/usr/bin/env python
import os
import sys
import json
from PIL import Image
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration
from src.explainability.reasoning.visual_context_extractor import VisualContextExtractor
from src.explainability.reasoning.query_reformulator import QueryReformulator

def main():
    print("=== Simple Query Reformulation Test ===")
    
    # Load config
    config = Config('configs/config.yaml')
    
    # Setup logger
    logger = setup_logger('simple_test', config['logging']['save_dir'], level='INFO')
    
    # Create output directory
    output_dir = 'data/query_reformulation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load sample from test data
        test_questions_path = config['data']['test_questions']
        test_images_dir = config['data']['test_images']
        
        print(f"Reading from: {test_questions_path}")
        
        # Get first 3 samples
        samples = []
        with open(test_questions_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                samples.append(json.loads(line))
        
        print(f"Loaded {len(samples)} samples")
        
        # Initialize components
        logger.info("Initializing components...")
        blip_model = BLIP2VQA(config, train_mode=False)
        
        # Load model
        model_path = 'checkpoints/blip/checkpoints/best_hf_model'
        if os.path.isdir(model_path):
            blip_model.model = type(blip_model.model).from_pretrained(model_path)
            blip_model.model.to(blip_model.device)
        blip_model.model.eval()
        
        gemini = GeminiIntegration(config)
        visual_extractor = VisualContextExtractor(blip_model, config)
        query_reformulator = QueryReformulator(gemini, visual_extractor, config)
        
        logger.info("Components initialized")
        
        # Store results
        all_results = []
        
        # Test each sample
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            image_id = sample['image_id']
            question = sample['question']
            
            print(f"Image ID: {image_id}")
            print(f"Original Question: {question}")
            
            # Find image
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = Path(test_images_dir) / f"{image_id}{ext}"
                if test_path.exists():
                    image_path = test_path
                    break
            
            if not image_path:
                print(f"❌ Image not found for {image_id}")
                continue
            
            print(f"Image found: {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Test visual context extraction
            print("Testing visual context extraction...")
            visual_context = visual_extractor.extract_complete_context(image, question)
            print(f"Visual Description: {visual_context['visual_description']}")
            print(f"Anatomical Context: {visual_context['anatomical_context']}")
            
            # Test query reformulation
            print("Testing query reformulation...")
            result = query_reformulator.reformulate_question(image, question)
            
            print(f"Reformulated: {result['reformulated_question']}")
            print(f"Quality Score: {result['reformulation_quality']['score']:.3f}")
            print(f"Success: {result['success']}")
            
            # Store result
            result_data = {
                'sample_index': i + 1,
                'image_id': image_id,
                'image_path': str(image_path),
                'original_question': question,
                'original_answer': sample.get('answer', ''),
                'visual_context': {
                    'description': visual_context['visual_description'],
                    'anatomical_context': visual_context['anatomical_context']
                },
                'reformulated_question': result['reformulated_question'],
                'question_type': result['question_type'],
                'quality_score': result['reformulation_quality']['score'],
                'quality_details': result['reformulation_quality'],
                'success': result['success']
            }
            
            all_results.append(result_data)
        
        # Save results to file
        results_file = os.path.join(output_dir, 'query_reformulation_test_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Create summary
        successful = sum(1 for r in all_results if r['success'])
        avg_quality = sum(r['quality_score'] for r in all_results) / len(all_results) if all_results else 0
        
        summary = {
            'test_summary': {
                'total_samples': len(all_results),
                'successful_reformulations': successful,
                'average_quality_score': avg_quality,
                'success_rate': successful / len(all_results) if all_results else 0
            },
            'results': all_results
        }
        
        summary_file = os.path.join(output_dir, 'reformulation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Summary saved to: {summary_file}")
        
        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Total samples: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Success rate: {successful / len(all_results):.1%}")
        print(f"Average quality: {avg_quality:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
