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
import textwrap

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration

# ENHANCED: Import Chain-of-Thought components
from src.explainability.reasoning.visual_context_extractor import VisualContextExtractor
from src.explainability.reasoning.query_reformulator import QueryReformulator
from src.explainability.rationale.chain_of_thought import ChainOfThoughtGenerator
from src.explainability.grad_cam import GradCAM

def load_model(config, model_path, logger):
    """Tải mô hình BLIP đã trained"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        logger.info(f"Loading BLIP model from {model_path}")
        model = BLIP2VQA(config, train_mode=False)
        model.device = device
        
        if os.path.isdir(model_path):
            model.model = type(model.model).from_pretrained(model_path)
            model.model.to(device)
            logger.info("Loaded model from HuggingFace directory")
        else:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                model.model.load_state_dict(checkpoint)
        
        model.model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading BLIP model: {e}")
        return None

def load_test_samples(config, num_samples=1, random_seed=42):
    """Tải mẫu test ngẫu nhiên"""
    random.seed(random_seed)
    
    # Đường dẫn dữ liệu
    test_questions_file = config['data']['test_questions']
    test_images_dir = config['data']['test_images']
    
    # Tải danh sách câu hỏi
    questions = []
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                questions.append(item)
            except:
                continue
    
    # Chọn ngẫu nhiên
    selected_questions = random.sample(questions, min(num_samples, len(questions)))
    
    # Tìm đường dẫn hình ảnh
    samples = []
    for item in selected_questions:
        image_id = item['image_id']
        
        # Thử các phần mở rộng phổ biến
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
    
    return samples

def initialize_explainable_components(config, blip_model, logger):
    """
    FIXED: Initialize all explainable AI components với proper GradCAM setup
    
    Returns:
        Dict with all initialized components or None if critical failure
    """
    components = {}
    
    try:
        # Gemini Integration (CRITICAL)
        logger.info("Initializing Gemini Integration...")
        components['gemini'] = GeminiIntegration(config)
        logger.info("✅ Gemini Integration ready")
        
        # Visual Context Extractor  
        logger.info("Initializing Visual Context Extractor...")
        components['visual_extractor'] = VisualContextExtractor(blip_model, config)
        logger.info("✅ Visual Context Extractor ready")
        
        # Query Reformulator
        logger.info("Initializing Query Reformulator...")
        components['query_reformulator'] = QueryReformulator(
            components['gemini'], 
            components['visual_extractor'], 
            config
        )
        logger.info("✅ Query Reformulator ready")
        
        # FIXED: Grad-CAM with proper model setup
        logger.info("Initializing Grad-CAM...")
        try:
            # Ensure blip_model.model has processor attribute for GradCAM
            if not hasattr(blip_model.model, 'processor'):
                blip_model.model.processor = blip_model.processor
                logger.debug("Added processor attribute to model for GradCAM compatibility")
            
            components['grad_cam'] = GradCAM(blip_model.model, layer_name="vision_model.encoder.layers.11")
            logger.info("✅ Grad-CAM ready")
        except Exception as e:
            logger.warning(f"Grad-CAM initialization failed: {e}. Continuing without Grad-CAM.")
            components['grad_cam'] = None
        
        # Chain-of-Thought Generator
        logger.info("Initializing Chain-of-Thought Generator...")
        components['cot_generator'] = ChainOfThoughtGenerator(components['gemini'], config)
        logger.info("✅ Chain-of-Thought Generator ready")
        
        logger.info("🎉 All explainable AI components initialized successfully")
        return components
        
    except Exception as e:
        logger.error(f"❌ Critical error initializing explainable components: {e}")
        return None

def process_basic_vqa(blip_model, gemini, sample, logger):
    """
    PRESERVED: Basic VQA processing (original functionality)
    """
    image_path = sample['image_path']
    question = sample['question']
    ground_truth = sample['answer']
    
    # Tải hình ảnh
    image = Image.open(image_path).convert('RGB')
    
    # Dự đoán với BLIP
    logger.info(f"Processing image {sample['image_id']}")
    blip_answer = blip_model.predict(image, question)
    logger.info(f"Initial BLIP answer: {blip_answer}")
    
    # Tạo câu trả lời thống nhất
    logger.info("Generating unified answer...")
    unified_answer = gemini.generate_unified_answer(image, question, blip_answer)
    logger.info(f"Unified answer generated")
    
    return {
        'mode': 'basic_vqa',
        'image': image,
        'image_path': image_path,
        'question': question,
        'ground_truth': ground_truth,
        'blip_answer': blip_answer,
        'unified_answer': unified_answer,
        'processing_steps': [
            'BLIP inference',
            'Gemini enhancement'
        ],
        'success': True,
        'error_messages': []
    }

def process_explainable_vqa(blip_model, components, sample, enable_cot, logger):
    """
    ENHANCED: Explainable VQA processing với improved Chain-of-Thought integration
    """
    image_path = sample['image_path']
    question = sample['question']  
    ground_truth = sample['answer']
    
    # Tải hình ảnh
    image = Image.open(image_path).convert('RGB')
    
    logger.info(f"🔬 Processing explainable VQA for image {sample['image_id']}")
    
    # Initialize result structure
    result = {
        'mode': 'explainable_vqa',
        'chain_of_thought_enabled': enable_cot,
        'image': image,
        'image_path': image_path,
        'question': question,
        'ground_truth': ground_truth,
        'success': True,
        'error_messages': [],
        'processing_steps': []
    }
    
    try:
        # Step 1: BLIP prediction
        logger.info("Step 1: BLIP inference...")
        blip_answer = blip_model.predict(image, question)
        result['blip_answer'] = blip_answer
        result['processing_steps'].append('BLIP inference')
        logger.info(f"✅ BLIP answer: {blip_answer}")
        
        # Step 2: Query Reformulation
        logger.info("Step 2: Query reformulation...")
        reformulation_result = components['query_reformulator'].reformulate_question(image, question)
        reformulated_question = reformulation_result['reformulated_question']
        visual_context = reformulation_result['visual_context']
        reformulation_quality = reformulation_result['reformulation_quality']['score']
        
        result['reformulated_question'] = reformulated_question
        result['reformulation_quality'] = reformulation_quality
        result['visual_context'] = visual_context
        result['processing_steps'].append('Query reformulation')
        logger.info(f"✅ Query reformulated (quality: {reformulation_quality:.3f})")
        
        # Step 3: Grad-CAM generation (FIXED)
        logger.info("Step 3: Grad-CAM attention analysis...")
        grad_cam_heatmap = None
        grad_cam_data = {}
        
        if components['grad_cam'] is not None:
            try:
                # FIXED: Use proper GradCAM call method
                grad_cam_heatmap = components['grad_cam'](
                    image, question, 
                    inputs=None,  # Let GradCAM handle input processing
                    original_size=image.size
                )
                
                if grad_cam_heatmap is not None:
                    # IMPROVED: Better region extraction from heatmap
                    grad_cam_data = {
                        'heatmap': grad_cam_heatmap,
                        'regions': extract_attention_regions(grad_cam_heatmap, image.size)
                    }
                    logger.info("✅ Grad-CAM generated successfully")
                else:
                    logger.warning("⚠️ Grad-CAM returned None")
                    result['error_messages'].append("Grad-CAM generation returned None")
                    
            except Exception as e:
                logger.error(f"❌ Grad-CAM error: {e}")
                result['error_messages'].append(f"Grad-CAM error: {str(e)}")
                import traceback
                logger.debug(f"Grad-CAM traceback: {traceback.format_exc()}")
        else:
            logger.warning("⚠️ Grad-CAM not available")
            result['error_messages'].append("Grad-CAM component not initialized")
        
        result['grad_cam_heatmap'] = grad_cam_heatmap
        result['processing_steps'].append('Grad-CAM attention')
        
        # Step 4: Chain-of-Thought reasoning (if enabled)
        reasoning_result = None
        if enable_cot:
            logger.info("Step 4: Chain-of-Thought reasoning...")
            try:
                reasoning_result = components['cot_generator'].generate_reasoning_chain(
                    image=image,
                    reformulated_question=reformulated_question,
                    blip_answer=blip_answer,
                    visual_context=visual_context,
                    grad_cam_data=grad_cam_data
                )
                
                if reasoning_result['success']:
                    reasoning_confidence = reasoning_result['reasoning_chain']['overall_confidence']
                    reasoning_flow = reasoning_result['reasoning_chain']['flow_type']
                    step_count = len(reasoning_result['reasoning_chain']['steps'])
                    
                    logger.info(f"✅ Chain-of-Thought generated (flow: {reasoning_flow}, confidence: {reasoning_confidence:.3f}, steps: {step_count})")
                else:
                    logger.error(f"❌ Chain-of-Thought failed: {reasoning_result.get('error', 'Unknown error')}")
                    result['error_messages'].append(f"Chain-of-Thought failed: {reasoning_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ Chain-of-Thought error: {e}")
                result['error_messages'].append(f"Chain-of-Thought error: {str(e)}")
                reasoning_result = None
            
            result['processing_steps'].append('Chain-of-Thought reasoning')
        
        result['reasoning_result'] = reasoning_result
        
        # Step 5: Unified answer generation
        logger.info("Step 5: Final unified answer generation...")
        
        # IMPROVED: Enhanced context for unified answer
        enhanced_context = None
        if reasoning_result and reasoning_result['success']:
            # Extract conclusion from Chain-of-Thought
            reasoning_steps = reasoning_result['reasoning_chain']['steps']
            conclusion_step = next((step for step in reasoning_steps if step['type'] == 'conclusion'), None)
            
            if conclusion_step:
                enhanced_context = f"Chain-of-thought conclusion: {conclusion_step['content']}"
            else:
                # Use all steps summary
                step_summaries = [f"{step['type']}: {step['content'][:100]}..." for step in reasoning_steps[:3]]
                enhanced_context = "Chain-of-thought analysis: " + " | ".join(step_summaries)
        
        # Generate unified answer
        unified_answer = components['gemini'].generate_unified_answer(
            image, reformulated_question, blip_answer, 
            heatmap=grad_cam_heatmap,
            region_descriptions=enhanced_context
        )
        
        result['unified_answer'] = unified_answer
        result['processing_steps'].append('Unified answer generation')
        logger.info("✅ Explainable VQA processing completed")
        
    except Exception as e:
        logger.error(f"❌ Critical error in explainable VQA processing: {e}")
        result['success'] = False
        result['error_messages'].append(f"Critical processing error: {str(e)}")
        result['unified_answer'] = f"Processing failed: {str(e)}"
    
    return result

def extract_attention_regions(heatmap, image_size, threshold=0.5):
    """
    IMPROVED: Extract attention regions from Grad-CAM heatmap
    
    Args:
        heatmap: Numpy array heatmap
        image_size: (width, height) of original image
        threshold: Attention threshold for region detection
        
    Returns:
        List of region dictionaries
    """
    import numpy as np
    
    try:
        if heatmap is None:
            return []
        
        # Find high-attention areas
        high_attention = heatmap > threshold
        
        # Simple region extraction - find contours or connected components
        # For now, use a simple approach with peak detection
        try:
            from scipy import ndimage
            
            # Find local maxima
            local_maxima = ndimage.maximum_filter(heatmap, size=5) == heatmap
            peaks = np.where(local_maxima & (heatmap > threshold))
            
            regions = []
            for i in range(len(peaks[0])):
                y, x = peaks[0][i], peaks[1][i]
                score = heatmap[y, x]
                
                # Convert to original image coordinates
                scale_x = image_size[0] / heatmap.shape[1]
                scale_y = image_size[1] / heatmap.shape[0]
                
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                
                # Create region with reasonable size
                region_size = max(20, int(min(image_size) * 0.1))
                
                regions.append({
                    'bbox': [orig_x - region_size//2, orig_y - region_size//2, region_size, region_size],
                    'score': float(score),
                    'center': [orig_x, orig_y]
                })
            
            # Sort by attention score and return top regions
            regions.sort(key=lambda x: x['score'], reverse=True)
            return regions[:5]  # Return top 5 regions
            
        except ImportError:
            # Fallback without scipy
            # Simple peak detection
            max_val = np.max(heatmap)
            peak_locations = np.where(heatmap > max_val * 0.8)
            
            regions = []
            for i in range(min(5, len(peak_locations[0]))):  # Limit to 5 peaks
                y, x = peak_locations[0][i], peak_locations[1][i]
                score = heatmap[y, x]
                
                # Convert to original image coordinates
                scale_x = image_size[0] / heatmap.shape[1]
                scale_y = image_size[1] / heatmap.shape[0]
                
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                region_size = max(20, int(min(image_size) * 0.1))
                
                regions.append({
                    'bbox': [orig_x - region_size//2, orig_y - region_size//2, region_size, region_size],
                    'score': float(score),
                    'center': [orig_x, orig_y]
                })
            
            return regions
        
    except Exception as e:
        print(f"Error extracting attention regions: {e}")
        return []

def create_visualization(result, output_dir, logger):
    """
    ENHANCED: Create visualization với improved layout và error handling
    """
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    mode = result['mode']
    image = result['image']
    sample_id = Path(result['image_path']).stem
    success = result['success']
    
    try:
        if mode == 'basic_vqa':
            # Basic visualization (2x1 layout)
            fig = plt.figure(figsize=(12, 6))
            
            # Image
            ax_image = plt.subplot(1, 2, 1)
            ax_image.imshow(image)
            ax_image.set_title(f"MedXplain-VQA: {sample_id}", fontsize=12)
            ax_image.axis('off')
            
            # Text
            ax_text = plt.subplot(1, 2, 2)
            text_content = (
                f"Question: {result['question']}\n\n"
                f"Ground truth: {result['ground_truth']}\n\n"
                f"MedXplain-VQA answer: {result['unified_answer']}"
            )
            
            if not success:
                text_content += f"\n\nErrors: {'; '.join(result['error_messages'])}"
            
            ax_text.text(0.01, 0.99, text_content, transform=ax_text.transAxes,
                        fontsize=10, verticalalignment='top', wrap=True)
            ax_text.axis('off')
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, f"medxplain_basic_{sample_id}.png")
            
        else:  # explainable_vqa mode
            # Enhanced visualization
            enable_cot = result['chain_of_thought_enabled']
            
            if enable_cot:
                # 2x3 layout for full explainable pipeline
                fig = plt.figure(figsize=(18, 12))
                
                # Original image
                ax_image = plt.subplot2grid((2, 3), (0, 0))
                ax_image.imshow(image)
                ax_image.set_title("Original Image", fontsize=12)
                ax_image.axis('off')
                
                # Grad-CAM heatmap
                ax_heatmap = plt.subplot2grid((2, 3), (0, 1))
                if result['grad_cam_heatmap'] is not None:
                    ax_heatmap.imshow(result['grad_cam_heatmap'], cmap='jet')
                    ax_heatmap.set_title("Attention Heatmap", fontsize=12)
                else:
                    ax_heatmap.text(0.5, 0.5, "Heatmap not available", ha='center', va='center')
                    ax_heatmap.set_title("Attention Heatmap (N/A)", fontsize=12)
                ax_heatmap.axis('off')
                
                # Chain-of-Thought summary
                ax_cot = plt.subplot2grid((2, 3), (0, 2))
                if result['reasoning_result'] and result['reasoning_result']['success']:
                    reasoning_chain = result['reasoning_result']['reasoning_chain']
                    steps = reasoning_chain['steps']
                    confidence = reasoning_chain['overall_confidence']
                    
                    cot_text = f"Chain-of-Thought Reasoning\n"
                    cot_text += f"Flow: {reasoning_chain['flow_type']}\n"
                    cot_text += f"Confidence: {confidence:.3f}\n"
                    cot_text += f"Steps: {len(steps)}\n\n"
                    
                    # Show first 3 steps briefly
                    for i, step in enumerate(steps[:3]):
                        step_content = step['content'][:80] + "..." if len(step['content']) > 80 else step['content']
                        cot_text += f"{i+1}. {step['type']}: {step_content}\n\n"
                    
                    if len(steps) > 3:
                        cot_text += f"... and {len(steps)-3} more steps"
                else:
                    cot_text = "Chain-of-Thought reasoning\nnot available or failed"
                    if result.get('reasoning_result') and not result['reasoning_result']['success']:
                        cot_text += f"\nError: {result['reasoning_result'].get('error', 'Unknown')}"
                
                ax_cot.text(0.01, 0.99, cot_text, transform=ax_cot.transAxes,
                           fontsize=9, verticalalignment='top', wrap=True)
                ax_cot.set_title("Reasoning Chain", fontsize=12)
                ax_cot.axis('off')
                
                # Main text area (full width)
                ax_text = plt.subplot2grid((2, 3), (1, 0), colspan=3)
                
            else:
                # 2x2 layout for basic explainable (no Chain-of-Thought)
                fig = plt.figure(figsize=(15, 10))
                
                # Original image
                ax_image = plt.subplot2grid((2, 2), (0, 0))
                ax_image.imshow(image)
                ax_image.set_title("Original Image", fontsize=12)
                ax_image.axis('off')
                
                # Grad-CAM heatmap
                ax_heatmap = plt.subplot2grid((2, 2), (0, 1))
                if result['grad_cam_heatmap'] is not None:
                    ax_heatmap.imshow(result['grad_cam_heatmap'], cmap='jet')
                    ax_heatmap.set_title("Attention Heatmap", fontsize=12)
                else:
                    ax_heatmap.text(0.5, 0.5, "Heatmap not available", ha='center', va='center')
                    ax_heatmap.set_title("Attention Heatmap (N/A)", fontsize=12)
                ax_heatmap.axis('off')
                
                # Main text area
                ax_text = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            
            # Common text content for explainable mode
            text_content = f"Question: {result['question']}\n\n"
            text_content += f"Reformulated: {result['reformulated_question']}\n\n"
            text_content += f"Ground truth: {result['ground_truth']}\n\n"
            text_content += f"MedXplain-VQA answer: {result['unified_answer']}\n\n"
            text_content += f"Processing: {' → '.join(result['processing_steps'])}\n"
            text_content += f"Reformulation quality: {result['reformulation_quality']:.3f}"
            
            if enable_cot and result['reasoning_result'] and result['reasoning_result']['success']:
                confidence = result['reasoning_result']['reasoning_chain']['overall_confidence']
                text_content += f" | Reasoning confidence: {confidence:.3f}"
            
            # Add error information if any
            if result['error_messages']:
                text_content += f"\n\nIssues encountered: {'; '.join(result['error_messages'])}"
            
            ax_text.text(0.01, 0.99, text_content, transform=ax_text.transAxes,
                        fontsize=10, verticalalignment='top', wrap=True)
            ax_text.axis('off')
            
            # Set title without unicode characters (fix font warning)
            mode_title = "Enhanced" if enable_cot else "Basic"
            success_indicator = "SUCCESS" if success else "WARNING"
            plt.suptitle(f"[{success_indicator}] MedXplain-VQA {mode_title} Explainable Analysis: {sample_id}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            mode_suffix = "enhanced" if enable_cot else "explainable"
            output_file = os.path.join(output_dir, f"medxplain_{mode_suffix}_{sample_id}.png")
        
        # Save visualization
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)
        logger.info(f"✅ Visualization saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error creating visualization: {e}")
        return None

def save_results_metadata(result, output_dir, logger):
    """Save detailed results metadata với improved structure"""
    try:
        sample_id = Path(result['image_path']).stem
        mode = result['mode']
        
        # Create metadata
        metadata = {
            'sample_id': sample_id,
            'processing_mode': mode,
            'success': result['success'],
            'image_path': result['image_path'],
            'question': result['question'],
            'ground_truth': result['ground_truth'],
            'blip_answer': result['blip_answer'],
            'unified_answer': result['unified_answer'],
            'processing_steps': result['processing_steps'],
            'error_messages': result.get('error_messages', [])
        }
        
        # Add mode-specific metadata
        if mode == 'explainable_vqa':
            metadata.update({
                'chain_of_thought_enabled': result['chain_of_thought_enabled'],
                'reformulated_question': result['reformulated_question'],
                'reformulation_quality': result['reformulation_quality'],
                'grad_cam_available': result['grad_cam_heatmap'] is not None
            })
            
            if result['reasoning_result'] and result['reasoning_result']['success']:
                reasoning_chain = result['reasoning_result']['reasoning_chain']
                validation = reasoning_chain.get('validation', {})
                
                reasoning_metadata = {
                    'reasoning_confidence': reasoning_chain['overall_confidence'],
                    'reasoning_flow': reasoning_chain['flow_type'],
                    'reasoning_steps_count': len(reasoning_chain['steps']),
                    'confidence_method': reasoning_chain.get('confidence_propagation', 'unknown'),
                    'validation_score': validation.get('combined_score', 0.0),
                    'validation_validity': validation.get('overall_validity', False)
                }
                metadata['reasoning_analysis'] = reasoning_metadata
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"medxplain_{mode}_{sample_id}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Metadata saved to {metadata_file}")
        return metadata_file
        
    except Exception as e:
        logger.error(f"❌ Error saving metadata: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Enhanced MedXplain-VQA with Chain-of-Thought')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to specific image (optional)')
    parser.add_argument('--question', type=str, default=None, help='Specific question (optional)')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of test samples (if no image specified)')
    parser.add_argument('--output-dir', type=str, default='data/medxplain_enhanced_results', help='Output directory')
    
    # ENHANCED: Processing mode options
    parser.add_argument('--mode', type=str, default='explainable', 
                      choices=['basic', 'explainable', 'enhanced'],
                      help='Processing mode: basic (BLIP+Gemini), explainable (+ Query reformulation + Grad-CAM), enhanced (+ Chain-of-Thought)')
    parser.add_argument('--enable-cot', action='store_true', 
                      help='Enable Chain-of-Thought reasoning (same as --mode enhanced)')
    
    args = parser.parse_args()
    
    # Determine final processing mode
    if args.enable_cot or args.mode == 'enhanced':
        processing_mode = 'enhanced'
        enable_cot = True
    elif args.mode == 'explainable':
        processing_mode = 'explainable'
        enable_cot = False
    else:  # basic mode
        processing_mode = 'basic'
        enable_cot = False
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('medxplain_vqa_enhanced', config['logging']['save_dir'], level='INFO')
    logger.info(f"🚀 Starting Enhanced MedXplain-VQA (mode: {processing_mode})")
    
    # Tải mô hình BLIP
    blip_model = load_model(config, args.model_path, logger)
    if blip_model is None:
        logger.error("❌ Failed to load BLIP model. Exiting.")
        return
    
    # Initialize components based on mode
    if processing_mode == 'basic':
        # Basic mode: only Gemini needed
        try:
            gemini = GeminiIntegration(config)
            components = None
            logger.info("✅ Basic mode: Gemini integration ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            return
    else:
        # Explainable/Enhanced mode: full component suite
        components = initialize_explainable_components(config, blip_model, logger)
        if components is None:
            logger.error("❌ Failed to initialize explainable components. Exiting.")
            return
        gemini = components['gemini']
    
    # Process samples
    if args.image and args.question:
        # Single custom sample
        sample = {
            'image_id': Path(args.image).stem,
            'question': args.question,
            'answer': "Unknown (custom input)",
            'image_path': args.image
        }
        samples = [sample]
    else:
        # Load test samples
        logger.info(f"📊 Loading {args.num_samples} test samples")
        samples = load_test_samples(config, args.num_samples)
        
        if not samples:
            logger.error("❌ No test samples found. Exiting.")
            return
    
    logger.info(f"🎯 Processing {len(samples)} samples in {processing_mode} mode")
    
    # Process each sample
    results = []
    successful_results = 0
    
    for i, sample in enumerate(samples):
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Processing sample {i+1}/{len(samples)}: {sample['image_id']}")
        logger.info(f"{'='*60}")
        
        try:
            if processing_mode == 'basic':
                # Basic VQA processing
                result = process_basic_vqa(blip_model, gemini, sample, logger)
            else:
                # Explainable VQA processing
                result = process_explainable_vqa(blip_model, components, sample, enable_cot, logger)
            
            # Create visualization
            vis_file = create_visualization(result, args.output_dir, logger)
            
            # Save metadata  
            metadata_file = save_results_metadata(result, args.output_dir, logger)
            
            # Add file paths to result
            result['visualization_file'] = vis_file
            result['metadata_file'] = metadata_file
            
            results.append(result)
            
            if result['success']:
                successful_results += 1
                logger.info(f"✅ Sample {sample['image_id']} processed successfully")
            else:
                logger.warning(f"⚠️ Sample {sample['image_id']} processed with issues")
            
        except Exception as e:
            logger.error(f"❌ Error processing sample {sample['image_id']}: {e}")
            continue
    
    # Clean up Grad-CAM hooks if needed
    if components and 'grad_cam' in components and components['grad_cam'] is not None:
        components['grad_cam'].remove_hooks()
        logger.info("🧹 Grad-CAM hooks cleaned up")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Enhanced MedXplain-VQA COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Mode: {processing_mode}")
    logger.info(f"Samples processed: {successful_results}/{len(samples)} successful")
    logger.info(f"Results saved to: {args.output_dir}")
    
    if results:
        # Print summary for first successful result
        first_successful = next((r for r in results if r['success']), None)
        if first_successful:
            logger.info(f"\n📊 SAMPLE RESULT SUMMARY:")
            logger.info(f"Question: {first_successful['question']}")
            logger.info(f"Answer: {first_successful['unified_answer'][:100]}...")
            logger.info(f"Processing steps: {' → '.join(first_successful['processing_steps'])}")
            
            if 'reformulation_quality' in first_successful:
                logger.info(f"Reformulation quality: {first_successful['reformulation_quality']:.3f}")
            
            if enable_cot and first_successful.get('reasoning_result'):
                reasoning = first_successful['reasoning_result']
                if reasoning['success']:
                    confidence = reasoning['reasoning_chain']['overall_confidence']
                    logger.info(f"Reasoning confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()