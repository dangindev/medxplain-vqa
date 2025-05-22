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
import traceback
from datetime import datetime
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration

# Import explainable components
from src.explainability.reasoning.visual_context_extractor import VisualContextExtractor
from src.explainability.reasoning.query_reformulator import QueryReformulator
from src.explainability.rationale.chain_of_thought import ChainOfThoughtGenerator
from src.explainability.grad_cam import GradCAM

class BatchTestMonitor:
    """Monitoring system cho batch testing"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        self.performance_metrics = {
            'total_samples': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'processing_times': [],
            'confidence_scores': [],
            'reformulation_qualities': [],
            'errors': [],
            'component_failures': {
                'blip': 0,
                'query_reformulation': 0,
                'grad_cam': 0,
                'chain_of_thought': 0,
                'gemini': 0
            }
        }
    
    def start_batch(self, num_samples):
        """Bắt đầu batch testing"""
        self.start_time = time.time()
        self.performance_metrics['total_samples'] = num_samples
        print(f"\n🚀 BATCH TESTING STARTED")
        print(f"Samples: {num_samples}")
        print(f"Start time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def log_sample_start(self, sample_idx, sample_id):
        """Log bắt đầu xử lý sample"""
        print(f"\n📋 SAMPLE {sample_idx + 1}: {sample_id}")
        print("-" * 40)
        return time.time()
    
    def log_sample_success(self, sample_start_time, result):
        """Log thành công sample"""
        processing_time = time.time() - sample_start_time
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['successful_samples'] += 1
        
        # Extract metrics
        if 'reformulation_quality' in result:
            self.performance_metrics['reformulation_qualities'].append(result['reformulation_quality'])
        
        if result.get('reasoning_result') and result['reasoning_result']['success']:
            confidence = result['reasoning_result']['reasoning_chain']['overall_confidence']
            self.performance_metrics['confidence_scores'].append(confidence)
        
        print(f"✅ SUCCESS ({processing_time:.1f}s)")
        
        # Store result
        self.results.append({
            'sample_id': Path(result['image_path']).stem,
            'success': True,
            'processing_time': processing_time,
            'mode': result['mode'],
            'error_messages': result.get('error_messages', [])
        })
    
    def log_sample_failure(self, sample_start_time, sample_id, error):
        """Log thất bại sample"""
        processing_time = time.time() - sample_start_time
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['failed_samples'] += 1
        self.performance_metrics['errors'].append(str(error))
        
        print(f"❌ FAILED ({processing_time:.1f}s): {str(error)[:50]}...")
        
        # Store result
        self.results.append({
            'sample_id': sample_id,
            'success': False,
            'processing_time': processing_time,
            'error': str(error)
        })
    
    def log_component_failure(self, component_name):
        """Log component failure"""
        if component_name in self.performance_metrics['component_failures']:
            self.performance_metrics['component_failures'][component_name] += 1
    
    def finish_batch(self):
        """Kết thúc batch testing"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print(f"\n" + "="*60)
        print(f"🏁 BATCH TESTING COMPLETED")
        print(f"Total time: {total_time:.1f}s")
        print(f"Success rate: {self.get_success_rate():.1f}%")
        print("="*60)
    
    def get_success_rate(self):
        """Tính success rate"""
        if self.performance_metrics['total_samples'] == 0:
            return 0.0
        return (self.performance_metrics['successful_samples'] / self.performance_metrics['total_samples']) * 100
    
    def get_average_processing_time(self):
        """Tính thời gian xử lý trung bình"""
        times = self.performance_metrics['processing_times']
        return np.mean(times) if times else 0.0
    
    def get_average_confidence(self):
        """Tính confidence trung bình"""
        scores = self.performance_metrics['confidence_scores']
        return np.mean(scores) if scores else 0.0
    
    def get_average_reformulation_quality(self):
        """Tính reformulation quality trung bình"""
        qualities = self.performance_metrics['reformulation_qualities']
        return np.mean(qualities) if qualities else 0.0
    
    def generate_report(self):
        """Tạo báo cáo chi tiết"""
        total_time = self.end_time - self.start_time if self.end_time else 0
        
        report = {
            'batch_summary': {
                'total_samples': self.performance_metrics['total_samples'],
                'successful_samples': self.performance_metrics['successful_samples'],
                'failed_samples': self.performance_metrics['failed_samples'],
                'success_rate': self.get_success_rate(),
                'total_batch_time': total_time,
                'average_processing_time': self.get_average_processing_time()
            },
            'quality_metrics': {
                'average_confidence': self.get_average_confidence(),
                'average_reformulation_quality': self.get_average_reformulation_quality(),
                'confidence_range': [min(self.performance_metrics['confidence_scores']), max(self.performance_metrics['confidence_scores'])] if self.performance_metrics['confidence_scores'] else [0, 0],
                'reformulation_range': [min(self.performance_metrics['reformulation_qualities']), max(self.performance_metrics['reformulation_qualities'])] if self.performance_metrics['reformulation_qualities'] else [0, 0]
            },
            'component_reliability': self.performance_metrics['component_failures'],
            'errors': self.performance_metrics['errors'][:10],  # Top 10 errors
            'individual_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

def load_model(config, model_path, logger):
    """Tải mô hình BLIP với error handling"""
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

def load_diverse_test_samples(config, num_samples=10, random_seed=42):
    """Tải diverse test samples để test stability"""
    random.seed(random_seed)
    
    # Đường dẫn dữ liệu
    test_questions_file = config['data']['test_questions']
    test_images_dir = config['data']['test_images']
    
    # Tải tất cả câu hỏi
    all_questions = []
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                all_questions.append(item)
            except:
                continue
    
    print(f"📊 Total available questions: {len(all_questions)}")
    
    # Phân loại câu hỏi theo type để đảm bảo diversity
    question_types = {}
    for item in all_questions:
        first_word = item['question'].split()[0].lower()
        if first_word not in question_types:
            question_types[first_word] = []
        question_types[first_word].append(item)
    
    print(f"📈 Question types found: {list(question_types.keys())[:10]}...")
    
    # Chọn đa dạng từ các loại câu hỏi khác nhau
    selected_questions = []
    samples_per_type = max(1, num_samples // len(question_types))
    
    for question_type, questions in question_types.items():
        if len(selected_questions) >= num_samples:
            break
        
        # Chọn ngẫu nhiên từ mỗi type
        type_samples = random.sample(questions, min(samples_per_type, len(questions)))
        selected_questions.extend(type_samples)
    
    # Nếu chưa đủ, chọn thêm ngẫu nhiên
    if len(selected_questions) < num_samples:
        remaining_questions = [q for q in all_questions if q not in selected_questions]
        additional_samples = random.sample(remaining_questions, min(num_samples - len(selected_questions), len(remaining_questions)))
        selected_questions.extend(additional_samples)
    
    # Giới hạn về số lượng yêu cầu
    selected_questions = selected_questions[:num_samples]
    
    # Tìm đường dẫn hình ảnh và validate
    samples = []
    for item in selected_questions:
        image_id = item['image_id']
        
        # Thử các phần mở rộng phổ biến
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = Path(test_images_dir) / f"{image_id}{ext}"
            if img_path.exists():
                # Verify image can be loaded
                try:
                    test_img = Image.open(img_path)
                    test_img.close()
                    
                    samples.append({
                        'image_id': image_id,
                        'question': item['question'],
                        'answer': item['answer'],
                        'image_path': str(img_path),
                        'question_type': item['question'].split()[0].lower()
                    })
                    break
                except Exception as e:
                    print(f"⚠️ Skipping corrupted image {image_id}: {e}")
                    continue
    
    print(f"✅ Selected {len(samples)} valid samples")
    
    # Print diversity summary
    type_counts = {}
    for sample in samples:
        q_type = sample['question_type']
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    print(f"📊 Sample diversity: {dict(list(type_counts.items())[:5])}...")
    
    return samples

def initialize_explainable_components(config, blip_model, logger, monitor):
    """Initialize components với error tracking"""
    components = {}
    
    try:
        # Gemini Integration (CRITICAL)
        logger.info("Initializing Gemini Integration...")
        try:
            components['gemini'] = GeminiIntegration(config)
            logger.info("✅ Gemini Integration ready")
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            monitor.log_component_failure('gemini')
            return None
        
        # Visual Context Extractor  
        logger.info("Initializing Visual Context Extractor...")
        try:
            components['visual_extractor'] = VisualContextExtractor(blip_model, config)
            logger.info("✅ Visual Context Extractor ready")
        except Exception as e:
            logger.error(f"❌ Visual Context Extractor failed: {e}")
            # Non-critical, continue
        
        # Query Reformulator
        logger.info("Initializing Query Reformulator...")
        try:
            components['query_reformulator'] = QueryReformulator(
                components['gemini'], 
                components.get('visual_extractor'), 
                config
            )
            logger.info("✅ Query Reformulator ready")
        except Exception as e:
            logger.error(f"❌ Query Reformulator failed: {e}")
            monitor.log_component_failure('query_reformulation')
        
        # Grad-CAM
        logger.info("Initializing Grad-CAM...")
        try:
            if not hasattr(blip_model.model, 'processor'):
                blip_model.model.processor = blip_model.processor
            
            components['grad_cam'] = GradCAM(blip_model.model, layer_name="vision_model.encoder.layers.11")
            logger.info("✅ Grad-CAM ready")
        except Exception as e:
            logger.warning(f"⚠️ Grad-CAM initialization failed: {e}. Continuing without Grad-CAM.")
            components['grad_cam'] = None
            monitor.log_component_failure('grad_cam')
        
        # Chain-of-Thought Generator
        logger.info("Initializing Chain-of-Thought Generator...")
        try:
            components['cot_generator'] = ChainOfThoughtGenerator(components['gemini'], config)
            logger.info("✅ Chain-of-Thought Generator ready")
        except Exception as e:
            logger.error(f"❌ Chain-of-Thought Generator failed: {e}")
            monitor.log_component_failure('chain_of_thought')
        
        logger.info("🎉 Component initialization completed")
        return components
        
    except Exception as e:
        logger.error(f"❌ Critical error initializing components: {e}")
        return None

def process_enhanced_sample(blip_model, components, sample, logger, monitor):
    """Process single sample với comprehensive error handling"""
    image_path = sample['image_path']
    question = sample['question']  
    ground_truth = sample['answer']
    
    # Tải hình ảnh
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"❌ Error loading image: {e}")
        raise Exception(f"Image loading failed: {e}")
    
    # Initialize result structure
    result = {
        'mode': 'enhanced_batch',
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
        logger.debug("Step 1: BLIP inference...")
        try:
            blip_answer = blip_model.predict(image, question)
            result['blip_answer'] = blip_answer
            result['processing_steps'].append('BLIP inference')
            logger.debug(f"✅ BLIP answer: {blip_answer}")
        except Exception as e:
            logger.error(f"❌ BLIP inference failed: {e}")
            monitor.log_component_failure('blip')
            raise Exception(f"BLIP inference failed: {e}")
        
        # Step 2: Query Reformulation
        logger.debug("Step 2: Query reformulation...")
        reformulated_question = question  # Default fallback
        reformulation_quality = 0.5  # Default
        visual_context = {}
        
        if 'query_reformulator' in components:
            try:
                reformulation_result = components['query_reformulator'].reformulate_question(image, question)
                reformulated_question = reformulation_result['reformulated_question']
                visual_context = reformulation_result['visual_context']
                reformulation_quality = reformulation_result['reformulation_quality']['score']
                
                result['reformulated_question'] = reformulated_question
                result['reformulation_quality'] = reformulation_quality
                result['visual_context'] = visual_context
                result['processing_steps'].append('Query reformulation')
                logger.debug(f"✅ Query reformulated (quality: {reformulation_quality:.3f})")
            except Exception as e:
                logger.warning(f"⚠️ Query reformulation failed: {e}")
                result['error_messages'].append(f"Query reformulation failed: {str(e)}")
                monitor.log_component_failure('query_reformulation')
        else:
            result['error_messages'].append("Query reformulator not available")
        
        # Step 3: Grad-CAM generation
        logger.debug("Step 3: Grad-CAM attention analysis...")
        grad_cam_heatmap = None
        grad_cam_data = {}
        
        if components.get('grad_cam') is not None:
            try:
                grad_cam_heatmap = components['grad_cam'](
                    image, question, 
                    inputs=None,
                    original_size=image.size
                )
                
                if grad_cam_heatmap is not None:
                    grad_cam_data = {
                        'heatmap': grad_cam_heatmap,
                        'regions': []  # Simplified for batch testing
                    }
                    logger.debug("✅ Grad-CAM generated successfully")
                else:
                    logger.warning("⚠️ Grad-CAM returned None")
                    result['error_messages'].append("Grad-CAM generation returned None")
                    
            except Exception as e:
                logger.warning(f"⚠️ Grad-CAM error: {e}")
                result['error_messages'].append(f"Grad-CAM error: {str(e)}")
                monitor.log_component_failure('grad_cam')
        else:
            result['error_messages'].append("Grad-CAM component not initialized")
        
        result['grad_cam_heatmap'] = grad_cam_heatmap
        result['processing_steps'].append('Grad-CAM attention')
        
        # Step 4: Chain-of-Thought reasoning
        logger.debug("Step 4: Chain-of-Thought reasoning...")
        reasoning_result = None
        
        if 'cot_generator' in components:
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
                    
                    logger.debug(f"✅ Chain-of-Thought generated (flow: {reasoning_flow}, confidence: {reasoning_confidence:.3f}, steps: {step_count})")
                else:
                    logger.warning(f"⚠️ Chain-of-Thought failed: {reasoning_result.get('error', 'Unknown error')}")
                    result['error_messages'].append(f"Chain-of-Thought failed: {reasoning_result.get('error', 'Unknown error')}")
                    monitor.log_component_failure('chain_of_thought')
                    
            except Exception as e:
                logger.warning(f"⚠️ Chain-of-Thought error: {e}")
                result['error_messages'].append(f"Chain-of-Thought error: {str(e)}")
                reasoning_result = None
                monitor.log_component_failure('chain_of_thought')
        else:
            result['error_messages'].append("Chain-of-Thought generator not available")
        
        result['reasoning_result'] = reasoning_result
        result['processing_steps'].append('Chain-of-Thought reasoning')
        
        # Step 5: Unified answer generation
        logger.debug("Step 5: Final unified answer generation...")
        
        try:
            # Enhanced context for unified answer
            enhanced_context = None
            if reasoning_result and reasoning_result['success']:
                reasoning_steps = reasoning_result['reasoning_chain']['steps']
                conclusion_step = next((step for step in reasoning_steps if step['type'] == 'conclusion'), None)
                
                if conclusion_step:
                    enhanced_context = f"Chain-of-thought conclusion: {conclusion_step['content']}"
            
            # Generate unified answer
            unified_answer = components['gemini'].generate_unified_answer(
                image, reformulated_question, blip_answer, 
                heatmap=grad_cam_heatmap,
                region_descriptions=enhanced_context
            )
            
            result['unified_answer'] = unified_answer
            result['processing_steps'].append('Unified answer generation')
            logger.debug("✅ Enhanced processing completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Unified answer generation failed: {e}")
            result['error_messages'].append(f"Unified answer generation failed: {str(e)}")
            result['unified_answer'] = blip_answer  # Fallback to BLIP answer
            monitor.log_component_failure('gemini')
        
    except Exception as e:
        logger.error(f"❌ Critical error in enhanced processing: {e}")
        result['success'] = False
        result['error_messages'].append(f"Critical processing error: {str(e)}")
        result['unified_answer'] = f"Processing failed: {str(e)}"
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Batch Testing for MedXplain-VQA Stability')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='data/batch_test_results', help='Output directory')
    parser.add_argument('--mode', type=str, default='enhanced', choices=['enhanced', 'explainable'],
                      help='Processing mode for batch testing')
    parser.add_argument('--save-visualizations', action='store_true', 
                      help='Save individual visualizations (slower)')
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('batch_test_medxplain', config['logging']['save_dir'], level='INFO')
    
    # Initialize monitoring
    monitor = BatchTestMonitor()
    monitor.start_batch(args.num_samples)
    
    # Load model
    blip_model = load_model(config, args.model_path, logger)
    if blip_model is None:
        logger.error("❌ Failed to load BLIP model. Exiting.")
        return
    
    # Initialize components
    components = initialize_explainable_components(config, blip_model, logger, monitor)
    if components is None:
        logger.error("❌ Failed to initialize components. Exiting.")
        return
    
    # Load diverse test samples
    logger.info(f"📊 Loading {args.num_samples} diverse test samples")
    samples = load_diverse_test_samples(config, args.num_samples)
    
    if not samples:
        logger.error("❌ No test samples found. Exiting.")
        return
    
    logger.info(f"🎯 Starting batch processing ({args.mode} mode)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each sample
    successful_results = []
    
    for i, sample in enumerate(samples):
        sample_start_time = monitor.log_sample_start(i, sample['image_id'])
        
        try:
            # Process sample
            if args.mode == 'enhanced':
                result = process_enhanced_sample(blip_model, components, sample, logger, monitor)
            else:
                # Simplified explainable mode for comparison
                result = process_enhanced_sample(blip_model, components, sample, logger, monitor)
                # Remove Chain-of-Thought from processing
                if 'reasoning_result' in result:
                    result['reasoning_result'] = None
            
            # Log success
            monitor.log_sample_success(sample_start_time, result)
            successful_results.append(result)
            
            # Optional: Save individual visualization
            if args.save_visualizations and result['success']:
                try:
                    # Create simple visualization
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.imshow(result['image'])
                    ax.set_title(f"Sample: {sample['image_id']}")
                    ax.axis('off')
                    
                    vis_path = os.path.join(args.output_dir, f"sample_{sample['image_id']}.png")
                    plt.savefig(vis_path, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"⚠️ Visualization save failed for {sample['image_id']}: {e}")
            
        except Exception as e:
            monitor.log_sample_failure(sample_start_time, sample['image_id'], e)
            logger.error(f"❌ Sample {sample['image_id']} failed: {str(e)[:100]}...")
            continue
    
    # Clean up
    if components.get('grad_cam'):
        components['grad_cam'].remove_hooks()
    
    # Finish monitoring
    monitor.finish_batch()
    
    # Generate comprehensive report
    report = monitor.generate_report()
    
    # Save report
    report_file = os.path.join(args.output_dir, f"batch_test_report_{args.mode}_{args.num_samples}samples.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n📊 BATCH TEST SUMMARY")
    print(f"Mode: {args.mode}")
    print(f"Success Rate: {report['batch_summary']['success_rate']:.1f}%")
    print(f"Average Processing Time: {report['batch_summary']['average_processing_time']:.1f}s")
    print(f"Average Confidence: {report['quality_metrics']['average_confidence']:.3f}")
    print(f"Average Reformulation Quality: {report['quality_metrics']['average_reformulation_quality']:.3f}")
    print(f"Report saved: {report_file}")
    
    # Component reliability summary
    print(f"\n🔧 COMPONENT RELIABILITY:")
    for component, failures in report['component_reliability'].items():
        reliability = ((args.num_samples - failures) / args.num_samples) * 100
        status = "✅" if reliability >= 80 else "⚠️" if reliability >= 50 else "❌"
        print(f"{status} {component}: {reliability:.1f}% ({failures} failures)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    success_rate = report['batch_summary']['success_rate']
    
    if success_rate >= 90:
        print("🚀 EXCELLENT: System is production-ready!")
    elif success_rate >= 75:
        print("✅ GOOD: Minor improvements needed")
    elif success_rate >= 50:
        print("⚠️ FAIR: Significant improvements needed")
    else:
        print("❌ POOR: Major fixes required")
    
    if report['component_reliability']['grad_cam'] > 2:
        print("- Consider Grad-CAM stability improvements")
    
    if report['component_reliability']['chain_of_thought'] > 1:
        print("- Review Chain-of-Thought error handling")
    
    if report['quality_metrics']['average_confidence'] < 0.6:
        print("- Investigate confidence calculation issues")

if __name__ == "__main__":
    main()
