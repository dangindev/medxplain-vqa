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
from src.explainability.grad_cam import GradCAM
from src.explainability.rationale.chain_of_thought import ChainOfThoughtGenerator

def main():
    print("=== Chain-of-Thought Reasoning Test ===")
    
    # Load config
    config = Config('configs/config.yaml')
    
    # Setup logger
    logger = setup_logger('test_chain_of_thought', config['logging']['save_dir'], level='INFO')
    
    # Create output directory
    output_dir = 'data/chain_of_thought_test'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load sample from test data
        test_questions_path = config['data']['test_questions']
        test_images_dir = config['data']['test_images']
        
        print(f"Reading from: {test_questions_path}")
        
        # Get first sample
        with open(test_questions_path, 'r') as f:
            sample = json.loads(f.readline())
        
        image_id = sample['image_id']
        question = sample['question']
        answer = sample['answer']
        
        print(f"Testing with: {image_id}")
        print(f"Question: {question}")
        print(f"Ground truth: {answer}")
        
        # Find and load image
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            test_path = Path(test_images_dir) / f"{image_id}{ext}"
            if test_path.exists():
                image_path = test_path
                break
        
        if not image_path:
            print(f"❌ Image not found for {image_id}")
            return
        
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Loaded image: {image_path}")
        
        # Initialize all components
        logger.info("Initializing components...")
        
        # BLIP model
        blip_model = BLIP2VQA(config, train_mode=False)
        model_path = 'checkpoints/blip/checkpoints/best_hf_model'
        if os.path.isdir(model_path):
            blip_model.model = type(blip_model.model).from_pretrained(model_path)
            blip_model.model.to(blip_model.device)
        blip_model.model.eval()
        
        # Other components
        gemini = GeminiIntegration(config)
        visual_extractor = VisualContextExtractor(blip_model, config)
        query_reformulator = QueryReformulator(gemini, visual_extractor, config)
        
        # Grad-CAM
        grad_cam = GradCAM(blip_model.model, layer_name="vision_model.encoder.layers.11")
        blip_model.model.processor = blip_model.processor
        
        # Chain-of-Thought Generator
        cot_generator = ChainOfThoughtGenerator(gemini, config)
        
        logger.info("All components initialized")
        
        # Step 1: Get BLIP answer
        print("\n--- Step 1: BLIP Inference ---")
        blip_answer = blip_model.predict(image, question)
        print(f"BLIP Answer: {blip_answer}")
        
        # Step 2: Query reformulation
        print("\n--- Step 2: Query Reformulation ---")
        reformulation_result = query_reformulator.reformulate_question(image, question)
        reformulated_question = reformulation_result['reformulated_question']
        print(f"Reformulated: {reformulated_question}")
        
        # Step 3: Grad-CAM generation
        print("\n--- Step 3: Grad-CAM Generation ---")
        grad_cam_heatmap = grad_cam(image, question, original_size=image.size)
        
        # Prepare grad_cam_data
        grad_cam_data = {}
        if grad_cam_heatmap is not None:
            print("✅ Grad-CAM generated successfully")
            # Mock regions data (in real scenario, this comes from visualization.py)
            grad_cam_data = {
                'heatmap': grad_cam_heatmap,
                'regions': [{
                    'bbox': [50, 50, 100, 100],
                    'score': 0.8,
                    'center': [100, 100]
                }]
            }
        else:
            print("⚠️ Grad-CAM generation failed, continuing without")
        
        # Step 4: Chain-of-Thought Generation
        print("\n--- Step 4: Chain-of-Thought Generation ---")
        visual_context = reformulation_result['visual_context']
        
        reasoning_result = cot_generator.generate_reasoning_chain(
            image=image,
            reformulated_question=reformulated_question,
            blip_answer=blip_answer,
            visual_context=visual_context,
            grad_cam_data=grad_cam_data
        )
        
        if reasoning_result['success']:
            print("✅ Chain-of-Thought generated successfully")
            
            # Display reasoning chain
            reasoning_chain = reasoning_result['reasoning_chain']
            steps = reasoning_chain['steps']
            
            print(f"\nReasoning Flow: {reasoning_chain['flow_type']}")
            print(f"Overall Confidence: {reasoning_chain['overall_confidence']:.3f}")
            print(f"Total Steps: {len(steps)}")
            
            print("\n=== REASONING STEPS ===")
            for i, step in enumerate(steps):
                print(f"\nStep {i+1}: {step['type']}")
                print(f"Content: {step['content']}")
                print(f"Confidence: {step.get('confidence', 0.0):.3f}")
                
                # Show evidence links if available
                if 'evidence_links' in step:
                    evidence = step['evidence_links']
                    for evidence_type, links in evidence.items():
                        if links:
                            print(f"  {evidence_type}: {len(links)} items")
            
            # Save results
            complete_result = {
                'test_metadata': {
                    'image_id': image_id,
                    'image_path': str(image_path),
                    'original_question': question,
                    'ground_truth': answer
                },
                'blip_answer': blip_answer,
                'reformulation_result': {
                    'reformulated_question': reformulated_question,
                    'quality_score': reformulation_result['reformulation_quality']['score']
                },
                'reasoning_result': reasoning_result,
                'validation_results': reasoning_result.get('reasoning_chain', {}).get('validation', {})
            }
            
            # Save to file
            results_file = os.path.join(output_dir, f'chain_of_thought_test_{image_id}.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(complete_result, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Complete results saved to: {results_file}")
            
            # Print validation summary
            validation = reasoning_result.get('reasoning_chain', {}).get('validation', {})
            if validation:
                print(f"\n=== VALIDATION SUMMARY ===")
                print(f"Overall Validity: {validation.get('overall_validity', False)}")
                print(f"Combined Score: {validation.get('combined_score', 0.0):.3f}")
                
                template_val = validation.get('template_validation', {})
                print(f"Template Completeness: {template_val.get('completeness_score', 0.0):.3f}")
                print(f"Template Consistency: {template_val.get('consistency_score', 0.0):.3f}")
                
                medical_val = validation.get('medical_validation', {})
                print(f"Medical Accuracy: {medical_val.get('medical_accuracy_score', 0.0):.3f}")
                print(f"Logical Consistency: {medical_val.get('logical_consistency_score', 0.0):.3f}")
        
        else:
            print(f"❌ Chain-of-Thought generation failed: {reasoning_result.get('error', 'Unknown error')}")
        
        # Clean up
        grad_cam.remove_hooks()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
