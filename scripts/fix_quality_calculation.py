#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import time
import numpy as np
from collections import defaultdict

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

def improved_quality_calculation(reformulation_result, cot_result, blip_answer, final_answer):
    """
    FIXED: Improved and consistent quality calculation method
    
    Args:
        reformulation_result: Query reformulation result
        cot_result: Chain-of-thought result (can be None)
        blip_answer: BLIP answer
        final_answer: Final enhanced answer
        
    Returns:
        dict: Quality metrics
    """
    
    quality_components = []
    weights = []
    
    # Component 1: Reformulation quality (always available)
    reformulation_quality = reformulation_result['reformulation_quality']['score']
    quality_components.append(reformulation_quality)
    weights.append(0.3)  # 30% weight
    
    # Component 2: Chain-of-thought quality (if available)
    cot_confidence = 0.0
    cot_validity = False
    
    if cot_result and cot_result.get('success', False):
        cot_confidence = cot_result['reasoning_chain'].get('overall_confidence', 0.0)
        validation = cot_result['reasoning_chain'].get('validation', {})
        cot_validity = validation.get('overall_validity', False)
        
        quality_components.append(cot_confidence)
        weights.append(0.4)  # 40% weight for reasoning confidence
        
        quality_components.append(1.0 if cot_validity else 0.5)  # Validity bonus/penalty
        weights.append(0.2)  # 20% weight for validity
    else:
        # No chain-of-thought: redistribute weights
        weights[0] = 0.5  # Increase reformulation weight to 50%
    
    # Component 3: Answer quality assessment (always available) 
    answer_quality = assess_answer_quality_improved(final_answer)
    remaining_weight = 1.0 - sum(weights)
    quality_components.append(answer_quality)
    weights.append(remaining_weight)
    
    # Calculate weighted average
    if len(quality_components) == len(weights) and sum(weights) > 0:
        overall_quality = sum(q * w for q, w in zip(quality_components, weights)) / sum(weights)
    else:
        # Fallback to simple average
        overall_quality = sum(quality_components) / len(quality_components)
    
    return {
        'reformulation_quality': reformulation_quality,
        'chain_of_thought_confidence': cot_confidence,
        'chain_of_thought_validity': cot_validity,
        'answer_quality': answer_quality,
        'overall_quality': overall_quality,
        'quality_components': quality_components,
        'weights_used': weights,
        'calculation_method': 'weighted_average_v2'
    }

def assess_answer_quality_improved(answer):
    """
    FIXED: Improved answer quality assessment
    
    Args:
        answer: Final answer string
        
    Returns:
        float: Quality score between 0 and 1
    """
    if not answer or len(answer.strip()) < 5:
        return 0.1  # Very low for empty/short answers
    
    answer_lower = answer.lower()
    
    # Medical terminology scoring (improved)
    medical_terms = {
        'high_value': ['pathology', 'diagnosis', 'histology', 'morphology', 'cellular', 'tissue'],
        'medium_value': ['clinical', 'examination', 'analysis', 'findings', 'features'],
        'low_value': ['image', 'shows', 'appears', 'visible', 'observed']
    }
    
    medical_score = 0.0
    for category, terms in medical_terms.items():
        term_count = sum(1 for term in terms if term in answer_lower)
        if category == 'high_value':
            medical_score += term_count * 0.15
        elif category == 'medium_value':
            medical_score += term_count * 0.10
        else:
            medical_score += term_count * 0.05
    
    medical_score = min(medical_score, 0.4)  # Cap at 0.4
    
    # Length and structure scoring (improved)
    length = len(answer)
    if length < 20:
        length_score = 0.1
    elif length < 50:
        length_score = 0.3
    elif length < 150:
        length_score = 0.6
    elif length < 300:
        length_score = 0.8
    else:
        length_score = 1.0
    
    # Coherence scoring (simple heuristic)
    sentences = answer.split('.')
    coherence_score = min(len([s for s in sentences if len(s.strip()) > 10]) / 5.0, 0.3)
    
    # Specificity scoring
    generic_terms = ['yes', 'no', 'maybe', 'unclear', 'unknown']
    specificity_penalty = sum(0.1 for term in generic_terms if term in answer_lower)
    specificity_score = max(0.0, 0.3 - specificity_penalty)
    
    # Combine scores
    total_score = medical_score + (length_score * 0.3) + coherence_score + specificity_score
    
    return min(total_score, 1.0)

def test_fixed_pipeline_sample(config, model_path, sample, output_dir, logger):
    """Test fixed pipeline on single sample"""
    
    # Initialize components
    logger.info("Initializing components with fixes...")
    
    blip_model = BLIP2VQA(config, train_mode=False)
    blip_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.isdir(model_path):
        blip_model.model = type(blip_model.model).from_pretrained(model_path)
        blip_model.model.to(blip_model.device)
    else:
        checkpoint = torch.load(model_path, map_location=blip_model.device)
        if 'model_state_dict' in checkpoint:
            blip_model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            blip_model.model.load_state_dict(checkpoint)
    
    blip_model.model.eval()
    
    # Other components
    gemini = GeminiIntegration(config)
    visual_extractor = VisualContextExtractor(blip_model, config)
    query_reformulator = QueryReformulator(gemini, visual_extractor, config)
    
    # FIXED Grad-CAM initialization
    grad_cam = GradCAM(blip_model, layer_name="vision_model.encoder.layers.11")  # Pass wrapper, not underlying model
    
    cot_generator = ChainOfThoughtGenerator(gemini, config)
    
    # Load image
    image = Image.open(sample['image_path']).convert('RGB')
    
    # Test both modes with FIXED quality calculation
    results = {}
    
    for mode in ['standard', 'chain_of_thought']:
        logger.info(f"Testing {mode} mode with FIXED quality calculation")
        
        # Step 1: BLIP inference
        blip_answer = blip_model.predict(image, sample['question'])
        
        # Step 2: Query reformulation
        reformulation_result = query_reformulator.reformulate_question(image, sample['question'])
        
        # Step 3: Grad-CAM (FIXED)
        grad_cam_data = None
        try:
            grad_cam_heatmap = grad_cam(image, sample['question'], original_size=image.size)
            if grad_cam_heatmap is not None:
                grad_cam_data = {
                    'heatmap': grad_cam_heatmap,
                    'regions': [{'bbox': [50, 50, 100, 100], 'score': 0.8, 'center': [100, 100]}]
                }
                logger.info(f"Grad-CAM generated successfully for {mode}")
            else:
                logger.warning(f"Grad-CAM generation failed for {mode}")
        except Exception as e:
            logger.error(f"Grad-CAM error in {mode}: {e}")
        
        # Step 4: Chain-of-thought (conditional)
        cot_result = None
        if mode == 'chain_of_thought':
            try:
                visual_context = reformulation_result['visual_context']
                cot_result = cot_generator.generate_reasoning_chain(
                    image=image,
                    reformulated_question=reformulation_result['reformulated_question'],
                    blip_answer=blip_answer,
                    visual_context=visual_context,
                    grad_cam_data=grad_cam_data
                )
                
                if cot_result['success']:
                    confidence = cot_result['reasoning_chain']['overall_confidence']
                    logger.info(f"Chain-of-thought generated with confidence: {confidence:.3f}")
                else:
                    logger.error(f"Chain-of-thought failed: {cot_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Chain-of-thought error: {e}")
                cot_result = {'success': False, 'error': str(e), 'reasoning_chain': {'overall_confidence': 0.0}}
        
        # Step 5: Final answer enhancement
        try:
            if mode == 'chain_of_thought' and cot_result and cot_result.get('success', False):
                reasoning_steps = cot_result['reasoning_chain']['steps']
                reasoning_summary = "\n".join([f"- {step['content'][:150]}..." if len(step['content']) > 150 
                                             else f"- {step['content']}" for step in reasoning_steps[:4]])
                
                final_answer = gemini.generate_unified_answer(
                    image, reformulation_result['reformulated_question'], blip_answer,
                    heatmap=grad_cam_data.get('heatmap') if grad_cam_data else None,
                    additional_context=f"Chain-of-thought reasoning:\n{reasoning_summary}"
                )
            else:
                final_answer = gemini.generate_unified_answer(
                    image, reformulation_result['reformulated_question'], blip_answer,
                    heatmap=grad_cam_data.get('heatmap') if grad_cam_data else None
                )
        except Exception as e:
            logger.error(f"Gemini enhancement error: {e}")
            final_answer = f"Enhanced analysis: {blip_answer}"
        
        # FIXED quality calculation
        quality_metrics = improved_quality_calculation(
            reformulation_result, cot_result, blip_answer, final_answer
        )
        
        results[mode] = {
            'blip_answer': blip_answer,
            'reformulated_question': reformulation_result['reformulated_question'],
            'final_answer': final_answer,
            'quality_metrics': quality_metrics,
            'cot_result': cot_result,
            'grad_cam_available': grad_cam_data is not None
        }
        
        logger.info(f"{mode} quality: {quality_metrics['overall_quality']:.3f}")
    
    # Clean up
    grad_cam.remove_hooks()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"FIXED_quality_test_{sample['image_id']}.json"), 'w') as f:
        json.dump({
            'sample': sample,
            'results': results,
            'comparison': {
                'quality_improvement': results['chain_of_thought']['quality_metrics']['overall_quality'] - 
                                     results['standard']['quality_metrics']['overall_quality'],
                'quality_ratio': results['chain_of_thought']['quality_metrics']['overall_quality'] / 
                               results['standard']['quality_metrics']['overall_quality'] 
                               if results['standard']['quality_metrics']['overall_quality'] > 0 else 0
            }
        }, f, indent=2, default=str)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test Fixed Quality Calculation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='data/quality_fix_test', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('quality_fix_test', config['logging']['save_dir'], level='INFO')
    logger.info("Testing FIXED quality calculation and Grad-CAM")
    
    # Load test samples
    test_questions_file = config['data']['test_questions']
    test_images_dir = config['data']['test_images']
    
    questions = []
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                questions.append(item)
            except:
                continue
    
    # Select random samples
    selected_samples = random.sample(questions, min(args.num_samples, len(questions)))
    
    samples = []
    for item in selected_samples:
        image_id = item['image_id']
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = Path(test_images_dir) / f"{image_id}{ext}"
            if img_path.exists():
                samples.append({
                    'image_id': image_id,
                    'question': item['question'],
                    'answer': item['answer'],
                    'image_path': str(img_path)
                })
                break
    
    if not samples:
        logger.error("No samples found")
        return
    
    # Test each sample
    all_results = []
    
    for sample in samples:
        logger.info(f"Testing sample: {sample['image_id']}")
        
        try:
            results = test_fixed_pipeline_sample(config, args.model_path, sample, args.output_dir, logger)
            all_results.append(results)
            
            # Print comparison
            std_quality = results['standard']['quality_metrics']['overall_quality']
            cot_quality = results['chain_of_thought']['quality_metrics']['overall_quality']
            improvement = cot_quality - std_quality
            
            logger.info(f"Quality comparison for {sample['image_id']}:")
            logger.info(f"  Standard: {std_quality:.3f}")
            logger.info(f"  Chain-of-Thought: {cot_quality:.3f}")
            logger.info(f"  Improvement: {improvement:+.3f}")
            
        except Exception as e:
            logger.error(f"Failed to test sample {sample['image_id']}: {e}")
    
    # Summary
    if all_results:
        std_qualities = [r['standard']['quality_metrics']['overall_quality'] for r in all_results]
        cot_qualities = [r['chain_of_thought']['quality_metrics']['overall_quality'] for r in all_results]
        
        logger.info("=== FIXED QUALITY CALCULATION SUMMARY ===")
        logger.info(f"Average Standard Quality: {np.mean(std_qualities):.3f} (±{np.std(std_qualities):.3f})")
        logger.info(f"Average Chain-of-Thought Quality: {np.mean(cot_qualities):.3f} (±{np.std(cot_qualities):.3f})")
        logger.info(f"Average Improvement: {np.mean(cot_qualities) - np.mean(std_qualities):+.3f}")
        logger.info(f"Improvement Ratio: {np.mean(cot_qualities) / np.mean(std_qualities):.2f}x")
    
    logger.info("FIXED quality calculation test completed")

if __name__ == "__main__":
    main()
