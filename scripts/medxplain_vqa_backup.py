#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import random
import textwrap

# 🔧 TORCH COMPATIBILITY FIX - Add missing get_default_device for older PyTorch versions
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# 🆕 NEW: Import Bounding Box components
from src.explainability.enhanced_grad_cam import EnhancedGradCAM
from src.explainability.bounding_box_extractor import BoundingBoxExtractor
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

def initialize_explainable_components(config, blip_model, enable_bbox, logger):
    """
    🆕 ENHANCED: Initialize explainable AI components với Bounding Box support
    
    Args:
        config: Configuration object
        blip_model: BLIP model instance
        enable_bbox: Enable bounding box extraction
        logger: Logger instance
        
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
        
        # 🔧 FIXED: Enhanced Grad-CAM initialization using working logic from test_bounding_box_system.py
        if enable_bbox:
            logger.info("🆕 Initializing Enhanced Grad-CAM with Bounding Boxes...")
            try:
                # Ensure model compatibility (same as test_bounding_box_system.py)
                if not hasattr(blip_model.model, 'processor'):
                    blip_model.model.processor = blip_model.processor
                    logger.debug("Added processor attribute for Enhanced Grad-CAM compatibility")
                
                # 🔧 FIXED: Use same bbox_config as working test_bounding_box_system.py
                bbox_config = {
                    'attention_threshold': 0.25,  # Lower threshold for better detection
                    'min_region_size': 6,
                    'max_regions': 5,
                    'box_expansion': 0.12
                }
                
                # Override with config if available
                if 'bounding_box' in config:
                    bbox_config.update(config['bounding_box'])
                
                # 🔧 FIXED: Initialize Enhanced Grad-CAM exactly like test_bounding_box_system.py
                components['enhanced_grad_cam'] = EnhancedGradCAM(
                    blip_model.model,  # Pass model directly (not with layer_name parameter)
                    bbox_config=bbox_config
                )
                
                # Initialize standalone BoundingBoxExtractor for utility functions
                components['bbox_extractor'] = BoundingBoxExtractor(bbox_config)
                
                logger.info("✅ Enhanced Grad-CAM with Bounding Boxes ready")
                components['grad_cam_mode'] = 'enhanced'
                
            except Exception as e:
                logger.error(f"❌ Enhanced Grad-CAM initialization failed: {e}")
                logger.info("Falling back to basic Grad-CAM...")
                enable_bbox = False
        
        # Basic Grad-CAM fallback
        if not enable_bbox:
            logger.info("Initializing Basic Grad-CAM...")
            try:
                if not hasattr(blip_model.model, 'processor'):
                    blip_model.model.processor = blip_model.processor
                
                components['grad_cam'] = GradCAM(blip_model.model, layer_name="vision_model.encoder.layers.11")
                logger.info("✅ Basic Grad-CAM ready")
                components['grad_cam_mode'] = 'basic'
                
            except Exception as e:
                logger.warning(f"Basic Grad-CAM initialization failed: {e}. Continuing without Grad-CAM.")
                components['grad_cam'] = None
                components['grad_cam_mode'] = 'none'
        
        # Chain-of-Thought Generator
        logger.info("Initializing Chain-of-Thought Generator...")
        components['cot_generator'] = ChainOfThoughtGenerator(components['gemini'], config)
        logger.info("✅ Chain-of-Thought Generator ready")
        
        # Set bounding box enabled flag
        components['bbox_enabled'] = enable_bbox
        
        logger.info(f"🎉 All explainable AI components initialized successfully (bbox_mode: {'enabled' if enable_bbox else 'disabled'})")
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
    🆕 ENHANCED: Explainable VQA processing với Bounding Box integration
    """
    image_path = sample['image_path']
    question = sample['question']  
    ground_truth = sample['answer']
    
    # Tải hình ảnh
    image = Image.open(image_path).convert('RGB')
    
    logger.info(f"🔬 Processing explainable VQA for image {sample['image_id']} (bbox: {components['bbox_enabled']})")
    
    # Initialize result structure
    result = {
        'mode': 'explainable_vqa',
        'chain_of_thought_enabled': enable_cot,
        'bbox_enabled': components['bbox_enabled'],
        'grad_cam_mode': components['grad_cam_mode'],
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
        
        # Step 3: 🆕 ENHANCED Grad-CAM generation with Bounding Boxes
        logger.info("Step 3: Enhanced Grad-CAM attention analysis...")
        grad_cam_heatmap = None
        grad_cam_data = {}
        bbox_regions = []
        
        if components['grad_cam_mode'] == 'enhanced':
            # 🆕 NEW: Enhanced Grad-CAM with Bounding Boxes
            try:
                enhanced_grad_cam = components['enhanced_grad_cam']
                
                logger.info("🆕 Running Enhanced Grad-CAM with bounding box extraction...")
                analysis_result = enhanced_grad_cam.analyze_image_with_question(
                    image, question, save_dir=None
                )
                
                if analysis_result['success']:
                    grad_cam_heatmap = analysis_result['heatmap']
                    bbox_regions = analysis_result['regions']
                    
                    grad_cam_data = {
                        'heatmap': grad_cam_heatmap,
                        'regions': bbox_regions,
                        'bbox_enabled': True
                    }
                    
                    logger.info(f"✅ Enhanced Grad-CAM generated: {len(bbox_regions)} bounding boxes detected")
                else:
                    logger.warning(f"⚠️ Enhanced Grad-CAM failed: {analysis_result.get('error', 'Unknown error')}")
                    result['error_messages'].append(f"Enhanced Grad-CAM error: {analysis_result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"❌ Enhanced Grad-CAM error: {e}")
                result['error_messages'].append(f"Enhanced Grad-CAM error: {str(e)}")
                
        elif components['grad_cam_mode'] == 'basic':
            # Fallback to basic Grad-CAM
            try:
                grad_cam = components['grad_cam']
                grad_cam_heatmap = grad_cam(image, question, original_size=image.size)
                
                if grad_cam_heatmap is not None:
                    # Extract basic attention regions
                    bbox_regions = extract_attention_regions_basic(grad_cam_heatmap, image.size)
                    
                    grad_cam_data = {
                        'heatmap': grad_cam_heatmap,
                        'regions': bbox_regions,
                        'bbox_enabled': False
                    }
                    logger.info(f"✅ Basic Grad-CAM generated: {len(bbox_regions)} attention regions detected")
                else:
                    logger.warning("⚠️ Basic Grad-CAM returned None")
                    result['error_messages'].append("Basic Grad-CAM generation returned None")
                    
            except Exception as e:
                logger.error(f"❌ Basic Grad-CAM error: {e}")
                result['error_messages'].append(f"Basic Grad-CAM error: {str(e)}")
        
        result['grad_cam_heatmap'] = grad_cam_heatmap
        result['bbox_regions'] = bbox_regions
        result['processing_steps'].append('Enhanced Grad-CAM attention')
        
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
        
        # Step 5: 🆕 ENHANCED Unified answer generation
        logger.info("Step 5: Enhanced unified answer generation...")
        
        # Prepare enhanced context
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
        
        # 🆕 ENHANCED: Add bounding box region descriptions
        region_descriptions = None
        if bbox_regions:
            region_descs = []
            for i, region in enumerate(bbox_regions[:3]):  # Top 3 regions
                bbox = region['bbox']
                score = region.get('attention_score', region.get('score', 0))
                region_descs.append(f"Region {i+1}: bbox {bbox} (attention: {score:.3f})")
            
            region_descriptions = "Attention regions: " + "; ".join(region_descs)
            
            if enhanced_context:
                enhanced_context += f" | {region_descriptions}"
            else:
                enhanced_context = region_descriptions
        
        # Generate unified answer with enhanced context
        unified_answer = components['gemini'].generate_unified_answer(
            image, reformulated_question, blip_answer, 
            heatmap=grad_cam_heatmap,
            region_descriptions=enhanced_context
        )
        
        result['unified_answer'] = unified_answer
        result['processing_steps'].append('Enhanced unified answer generation')
        logger.info("✅ Enhanced explainable VQA processing completed")
        
    except Exception as e:
        logger.error(f"❌ Critical error in explainable VQA processing: {e}")
        result['success'] = False
        result['error_messages'].append(f"Critical processing error: {str(e)}")
        result['unified_answer'] = f"Processing failed: {str(e)}"
    
    return result

def extract_attention_regions_basic(heatmap, image_size, threshold=0.5):
    """
    FALLBACK: Basic attention region extraction (when Enhanced Grad-CAM unavailable)
    """
    import numpy as np
    
    try:
        if heatmap is None:
            return []
        
        # Find high-attention areas
        high_attention = heatmap > threshold
        
        # Simple region extraction
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
                    'attention_score': float(score),  # For compatibility
                    'center': [orig_x, orig_y]
                })
            
            # Sort by attention score and return top regions
            regions.sort(key=lambda x: x['score'], reverse=True)
            return regions[:5]  # Return top 5 regions
            
        except ImportError:
            # Fallback without scipy
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
                    'attention_score': float(score),
                    'center': [orig_x, orig_y]
                })
            
            return regions
        
    except Exception as e:
        print(f"Error extracting basic attention regions: {e}")
        return []


def create_visualization(result, output_dir, logger):
    """
    🆕 ENHANCED: Create visualization với text đầy đủ, không cắt, căn giữa, chữ to
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path
    import os
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    mode = result['mode']
    image = result['image']
    sample_id = Path(result['image_path']).stem
    success = result['success']
    bbox_enabled = result.get('bbox_enabled', False)
    bbox_regions = result.get('bbox_regions', [])
    grad_cam_heatmap = result.get('grad_cam_heatmap', None)
    
    # 🔧 FIXED: Debug logging để kiểm tra các điều kiện
    logger.debug(f"Visualization debug - bbox_enabled: {bbox_enabled}, heatmap available: {grad_cam_heatmap is not None}, bbox_regions: {len(bbox_regions)}")
    
    try:
        if mode == 'basic_vqa':
            # Basic visualization (2x1 layout)
            fig = plt.figure(figsize=(14, 8))
            
            # Image
            ax_image = plt.subplot(1, 2, 1)
            ax_image.imshow(image)
            ax_image.set_title(f"MedXplain-VQA: {sample_id}", fontsize=14, fontweight='bold')
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
            
            ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes,
                        fontsize=12, verticalalignment='top', wrap=True,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.9))
            ax_text.axis('off')
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, f"medxplain_basic_{sample_id}.png")
            
        else:  # explainable_vqa mode
            enable_cot = result['chain_of_thought_enabled']
            
            # 🔧 FIXED: Improved condition để đảm bảo 4-panel layout khi có heatmap
            # Điều kiện: có heatmap VÀ (có bbox_enabled HOẶC có bbox_regions)
            has_heatmap = grad_cam_heatmap is not None
            has_bbox_data = bbox_enabled or len(bbox_regions) > 0
            
            logger.info(f"Visualization decision - has_heatmap: {has_heatmap}, has_bbox_data: {has_bbox_data}")
            
            if has_heatmap and has_bbox_data:
                # 🎯 IMPROVED: 4-panel layout với text area lớn hơn và chữ to
                logger.info("Creating 4-panel visualization with bounding boxes")
                fig = plt.figure(figsize=(32, 18))  # Lớn hơn để có space cho text
                
                # Create grid - tăng height ratio cho text area
                gs = fig.add_gridspec(3, 4, height_ratios=[4, 0.2, 1.5], hspace=0.3, wspace=0.1)
                
                # ===== 4 IMAGE PANELS (TOP ROW) =====
                
                # 1. Original Image
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(image)
                ax1.set_title('Original Image', fontsize=16, fontweight='bold', pad=20)
                ax1.axis('off')
                
                # 2. Bounding Boxes
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(image)
                ax2.set_title(f'Bounding Boxes ({len(bbox_regions)} regions)', fontsize=16, fontweight='bold', pad=20)
                ax2.axis('off')
                
                # Draw bounding boxes with improved styling
                colors = ['#FF0000', '#0066FF', '#00CC00', '#FFD700', '#8A2BE2', '#FF8C00', '#FF1493']
                for i, region in enumerate(bbox_regions[:5]):
                    bbox = region['bbox']
                    color = colors[i % len(colors)]
                    score = region.get('attention_score', region.get('score', 0))
                    
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=4, edgecolor=color, facecolor='none', alpha=0.9
                    )
                    ax2.add_patch(rect)
                    
                    # Improved label
                    ax2.text(
                        bbox[0], bbox[1] - 10,
                        f"R{i+1}: {score:.3f}",
                        color=color, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.95, 
                                edgecolor=color, linewidth=2)
                    )
                
                # 3. Heatmap Only
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(grad_cam_heatmap, cmap='jet', interpolation='bilinear')
                ax3.set_title('Enhanced Attention Heatmap', fontsize=16, fontweight='bold', pad=20)
                ax3.axis('off')
                
                # 4. Combined View
                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(image, alpha=0.65)
                ax4.imshow(grad_cam_heatmap, cmap='jet', alpha=0.35, interpolation='bilinear')
                ax4.set_title('Combined View', fontsize=16, fontweight='bold', pad=20)
                ax4.axis('off')
                
                # Draw bounding boxes on combined view
                for i, region in enumerate(bbox_regions[:5]):
                    bbox = region['bbox']
                    color = colors[i % len(colors)]
                    
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
                    )
                    ax4.add_patch(rect)
                
                # ===== TEXT AREA (BOTTOM ROW) - FULL WIDTH =====
                ax_text = fig.add_subplot(gs[2, :])
                
                # 🎯 IMPROVED TEXT CONTENT - NO TRUNCATION, LARGER FONT
                question_text = f"🔍 Question: {result['question']}"
                
                reformulated_text = f"📝 Reformulated: {result['reformulated_question']}"
                
                ground_truth_text = f"🎯 Ground Truth: {result['ground_truth']}"
                
                answer_text = f"🤖 MedXplain-VQA Answer: {result['unified_answer']}"
                
                processing_text = f"⚙️  Processing Pipeline: {' → '.join(result['processing_steps'])}"
                
                # Metrics line
                metrics_parts = []
                metrics_parts.append(f"Query Quality: {result['reformulation_quality']:.3f}")
                
                if bbox_regions:
                    avg_score = sum(r.get('attention_score', r.get('score', 0)) for r in bbox_regions) / len(bbox_regions)
                    metrics_parts.append(f"Bounding Boxes: {len(bbox_regions)} detected (avg: {avg_score:.3f})")
                
                if enable_cot and result['reasoning_result'] and result['reasoning_result']['success']:
                    confidence = result['reasoning_result']['reasoning_chain']['overall_confidence']
                    metrics_parts.append(f"Reasoning Confidence: {confidence:.3f}")
                
                metrics_text = f"📊 Metrics: {' | '.join(metrics_parts)}"
                
                # Combine all text
                all_text_lines = [
                    question_text,
                    "",
                    reformulated_text,
                    "",
                    ground_truth_text,
                    "",
                    answer_text,
                    "",
                    processing_text,
                    "",
                    metrics_text
                ]
                
                if result['error_messages']:
                    all_text_lines.extend(["", f"⚠️  Issues: {'; '.join(result['error_messages'])}"])
                
                full_text = '\n'.join(all_text_lines)
                
                # 🎯 ENHANCED TEXT DISPLAY - LARGER FONT, CENTERED, NO WRAP ISSUES
                ax_text.text(0.5, 0.5, full_text, 
                           transform=ax_text.transAxes,
                           fontsize=14,  # Increased from 12
                           verticalalignment='center',  # Center vertically
                           horizontalalignment='center',  # Center horizontally
                           wrap=True,
                           bbox=dict(boxstyle="round,pad=1.0", facecolor='#f8f9fa', 
                                   alpha=0.9, edgecolor='#dee2e6', linewidth=2))
                ax_text.axis('off')
                
                # Enhanced title
                success_indicator = "✅ SUCCESS" if success else "⚠️ WARNING"
                mode_title = "Enhanced" if enable_cot else "Explainable"
                plt.suptitle(f"{success_indicator} | MedXplain-VQA {mode_title} + BBox Analysis: {sample_id}", 
                           fontsize=20, fontweight='bold', y=0.98)
                
            else:
                # 🔧 FIXED: Fallback layout - cải thiện để hiển thị heatmap nếu có
                logger.info("Creating 2-panel fallback visualization")
                fig = plt.figure(figsize=(18, 12))
                
                ax_image = plt.subplot2grid((2, 2), (0, 0))
                ax_image.imshow(image)
                ax_image.set_title("Original Image", fontsize=16, fontweight='bold')
                ax_image.axis('off')
                
                ax_heatmap = plt.subplot2grid((2, 2), (0, 1))
                if grad_cam_heatmap is not None:
                    ax_heatmap.imshow(grad_cam_heatmap, cmap='jet')
                    ax_heatmap.set_title("Attention Heatmap", fontsize=16, fontweight='bold')
                else:
                    ax_heatmap.text(0.5, 0.5, "Heatmap not available", ha='center', va='center', fontsize=14)
                    ax_heatmap.set_title("Heatmap (N/A)", fontsize=16, fontweight='bold')
                ax_heatmap.axis('off')
                
                ax_text = plt.subplot2grid((2, 2), (1, 0), colspan=2)
                
                # Text content for fallback
                text_content = f"🔍 Question: {result['question']}\n\n"
                text_content += f"📝 Reformulated: {result['reformulated_question']}\n\n"
                text_content += f"🎯 Ground Truth: {result['ground_truth']}\n\n"
                text_content += f"🤖 MedXplain-VQA Answer: {result['unified_answer']}\n\n"
                text_content += f"⚙️ Processing: {' → '.join(result['processing_steps'])}\n"
                text_content += f"📊 Quality: {result['reformulation_quality']:.3f}"
                
                if result['error_messages']:
                    text_content += f"\n\n⚠️ Issues: {'; '.join(result['error_messages'])}"
                
                ax_text.text(0.5, 0.5, text_content, transform=ax_text.transAxes,
                           fontsize=14, verticalalignment='center', horizontalalignment='center',
                           wrap=True, bbox=dict(boxstyle="round,pad=1.0", facecolor='#f8f9fa', alpha=0.9))
                ax_text.axis('off')
                
                mode_title = "Enhanced" if enable_cot else "Explainable"
                success_indicator = "✅ SUCCESS" if success else "⚠️ WARNING"
                plt.suptitle(f"{success_indicator} | MedXplain-VQA {mode_title} Analysis: {sample_id}", 
                           fontsize=18, fontweight='bold')
            
            # File naming
            mode_suffix = "enhanced" if enable_cot else "explainable"
            bbox_suffix = "_bbox" if (bbox_enabled or len(bbox_regions) > 0) else ""
            output_file = os.path.join(output_dir, f"medxplain_{mode_suffix}{bbox_suffix}_{sample_id}.png")
        
        # Save with high quality
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.5, dpi=300, facecolor='white')
        plt.close(fig)
        logger.info(f"✅ Enhanced visualization saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error creating visualization: {e}")
        return None

    
def save_results_metadata(result, output_dir, logger):
    """🆕 ENHANCED: Save detailed results metadata với Bounding Box support"""
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
                'grad_cam_available': result['grad_cam_heatmap'] is not None,
                
                # 🆕 NEW: Bounding box metadata
                'bbox_enabled': result.get('bbox_enabled', False),
                'grad_cam_mode': result.get('grad_cam_mode', 'unknown'),
                'bbox_regions_count': len(result.get('bbox_regions', [])),
            })
            
            # 🆕 NEW: Detailed bounding box information
            bbox_regions = result.get('bbox_regions', [])
            if bbox_regions:
                bbox_metadata = {
                    'total_regions': len(bbox_regions),
                    'average_attention_score': sum(r.get('attention_score', r.get('score', 0)) for r in bbox_regions) / len(bbox_regions),
                    'max_attention_score': max(r.get('attention_score', r.get('score', 0)) for r in bbox_regions),
                    'regions_details': [
                        {
                            'rank': i + 1,
                            'bbox': region['bbox'],
                            'attention_score': region.get('attention_score', region.get('score', 0)),
                            'center': region.get('center', [0, 0])
                        }
                        for i, region in enumerate(bbox_regions[:5])  # Top 5 regions
                    ]
                }
                metadata['bounding_box_analysis'] = bbox_metadata
            
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
        bbox_suffix = "_bbox" if result.get('bbox_enabled', False) else ""
        metadata_file = os.path.join(output_dir, f"medxplain_{mode}{bbox_suffix}_{sample_id}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Enhanced metadata saved to {metadata_file}")
        return metadata_file
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced metadata: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='🆕 Enhanced MedXplain-VQA with Bounding Box Support')
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
    
    # 🆕 NEW: Bounding box support
    parser.add_argument('--enable-bbox', action='store_true', 
                      help='🆕 NEW: Enable bounding box extraction and visualization')
    
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
    
    bbox_status = "ENABLED" if args.enable_bbox else "DISABLED"
    logger.info(f"🚀 Starting Enhanced MedXplain-VQA (mode: {processing_mode}, bounding_boxes: {bbox_status})")
    
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
        # Explainable/Enhanced mode: full component suite with optional bounding boxes
        components = initialize_explainable_components(config, blip_model, args.enable_bbox, logger)
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
    
    bbox_mode = "with bounding boxes" if args.enable_bbox else "standard"
    logger.info(f"🎯 Processing {len(samples)} samples in {processing_mode} mode ({bbox_mode})")
    
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
    
    # Clean up hooks if needed
    if components:
        if 'enhanced_grad_cam' in components and components['enhanced_grad_cam'] is not None:
            components['enhanced_grad_cam'].grad_cam.remove_hooks()
            logger.info("🧹 Enhanced Grad-CAM hooks cleaned up")
        elif 'grad_cam' in components and components['grad_cam'] is not None:
            components['grad_cam'].remove_hooks()
            logger.info("🧹 Basic Grad-CAM hooks cleaned up")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Enhanced MedXplain-VQA COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Mode: {processing_mode} ({bbox_mode})")
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
            
            # 🆕 NEW: Bounding box summary
            if first_successful.get('bbox_regions'):
                bbox_count = len(first_successful['bbox_regions'])
                avg_score = sum(r.get('attention_score', r.get('score', 0)) for r in first_successful['bbox_regions']) / bbox_count
                logger.info(f"Bounding boxes: {bbox_count} detected (avg score: {avg_score:.3f})")
            
            if enable_cot and first_successful.get('reasoning_result'):
                reasoning = first_successful['reasoning_result']
                if reasoning['success']:
                    confidence = reasoning['reasoning_chain']['overall_confidence']
                    logger.info(f"Reasoning confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()

