import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class QuestionEnhancer:
    """
    Question Enhancement Pipeline that coordinates query reformulation process
    """
    
    def __init__(self, query_reformulator, config):
        """
        Initialize Question Enhancer
        
        Args:
            query_reformulator: QueryReformulator instance
            config: Configuration object
        """
        self.reformulator = query_reformulator
        self.config = config
        
        # Enhancement statistics
        self.enhancement_stats = {
            'total_processed': 0,
            'successful_reformulations': 0,
            'failed_reformulations': 0,
            'average_quality_score': 0.0
        }
        
        logger.info("Question Enhancer initialized")
    
    def enhance_single_question(self, image: Image.Image, question: str, 
                               save_intermediate: bool = False) -> Dict:
        """
        Enhance a single question with complete pipeline
        
        Args:
            image: PIL Image
            question: Original question
            save_intermediate: Save intermediate results
            
        Returns:
            Enhanced question result
        """
        logger.info(f"Enhancing question: {question}")
        
        try:
            # Step 1: Reformulate question
            reformulation_result = self.reformulator.reformulate_question(image, question)
            
            # Step 2: Validate enhancement
            validation_result = self._validate_enhancement(reformulation_result)
            
            # Step 3: Create final enhanced result
            enhanced_result = self._create_enhanced_result(
                reformulation_result, 
                validation_result
            )
            
            # Update statistics
            self._update_stats(enhanced_result)
            
            # Save intermediate results if requested
            if save_intermediate:
                self._save_intermediate_results(enhanced_result)
            
            logger.info("Question enhancement completed successfully")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error enhancing question: {e}")
            
            # Return error result
            return {
                'original_question': question,
                'enhanced_question': question,  # Fallback
                'enhancement_success': False,
                'error': str(e),
                'quality_metrics': {'overall_score': 0.0}
            }
    
    def _validate_enhancement(self, reformulation_result: Dict) -> Dict:
        """
        Validate the quality of question enhancement
        
        Args:
            reformulation_result: Result from query reformulation
            
        Returns:
            Validation result dictionary
        """
        validation = {
            'is_valid': False,
            'validation_score': 0.0,
            'validation_issues': [],
            'recommendations': []
        }
        
        if not reformulation_result['success']:
            validation['validation_issues'].append('reformulation_failed')
            return validation
        
        original = reformulation_result['original_question']
        reformulated = reformulation_result['reformulated_question']
        quality = reformulation_result['reformulation_quality']
        
        # Check quality score threshold
        if quality['score'] < 0.3:
            validation['validation_issues'].append('low_quality_score')
            validation['recommendations'].append('Consider manual review')
        
        # Check for sufficient enhancement
        if len(reformulated) <= len(original) * 1.1:
            validation['validation_issues'].append('insufficient_enhancement')
            validation['recommendations'].append('Add more image-specific details')
        
        # Check for image grounding
        if not quality.get('has_image_grounding', False):
            validation['validation_issues'].append('lacks_image_grounding')
            validation['recommendations'].append('Include image-specific references')
        
        # Check for medical terminology
        if not quality.get('has_medical_terms', False):
            validation['validation_issues'].append('lacks_medical_terminology')
            validation['recommendations'].append('Include relevant medical terms')
        
        # Compute overall validation score
        validation_score = quality['score']
        
        # Penalties for issues
        validation_score -= len(validation['validation_issues']) * 0.1
        validation_score = max(0.0, validation_score)
        
        validation['validation_score'] = validation_score
        validation['is_valid'] = validation_score >= 0.5 and len(validation['validation_issues']) <= 2
        
        return validation
    
    def _create_enhanced_result(self, reformulation_result: Dict, 
                               validation_result: Dict) -> Dict:
        """
        Create final enhanced question result
        
        Args:
            reformulation_result: Reformulation results
            validation_result: Validation results
            
        Returns:
            Complete enhanced result
        """
        enhanced_result = {
            # Original data
            'original_question': reformulation_result['original_question'],
            'enhanced_question': reformulation_result['reformulated_question'],
            
            # Enhancement metadata
            'enhancement_success': reformulation_result['success'] and validation_result['is_valid'],
            'question_type': reformulation_result.get('question_type', 'unknown'),
            
            # Quality metrics
            'quality_metrics': {
                'reformulation_score': reformulation_result['reformulation_quality']['score'],
                'validation_score': validation_result['validation_score'],
                'overall_score': (reformulation_result['reformulation_quality']['score'] + 
                                validation_result['validation_score']) / 2,
                'length_improvement': len(reformulation_result['reformulated_question']) / 
                                    len(reformulation_result['original_question'])
            },
            
            # Visual context
            'visual_context': {
                'description': reformulation_result.get('visual_context', {}).get('visual_description', ''),
                'anatomical_context': reformulation_result.get('visual_context', {}).get('anatomical_context', '')
            },
            
            # Issues and recommendations
            'validation_issues': validation_result['validation_issues'],
            'recommendations': validation_result['recommendations'],
            
            # Timestamp
            'timestamp': self._get_timestamp()
        }
        
        return enhanced_result
    
    def _update_stats(self, enhanced_result: Dict):
        """Update enhancement statistics"""
        self.enhancement_stats['total_processed'] += 1
        
        if enhanced_result['enhancement_success']:
            self.enhancement_stats['successful_reformulations'] += 1
        else:
            self.enhancement_stats['failed_reformulations'] += 1
        
        # Update average quality score
        current_avg = self.enhancement_stats['average_quality_score']
        new_score = enhanced_result['quality_metrics']['overall_score']
        total = self.enhancement_stats['total_processed']
        
        self.enhancement_stats['average_quality_score'] = (
            (current_avg * (total - 1) + new_score) / total
        )
    
    def _save_intermediate_results(self, enhanced_result: Dict):
        """Save intermediate enhancement results"""
        try:
            # Create output directory
            output_dir = Path(self.config.get('data.processed_dir', 'data/processed')) / 'query_reformulation'
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create filename based on original question
            safe_question = "".join(c for c in enhanced_result['original_question'][:50] 
                                   if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"enhanced_{safe_question.replace(' ', '_')}.json"
            
            output_path = output_dir / filename
            
            # Save result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_result, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Intermediate results saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    def enhance_dataset_questions(self, dataset_path: str, output_dir: str, 
                                 max_samples: Optional[int] = None) -> Dict:
        """
        Enhance questions from a dataset file
        
        Args:
            dataset_path: Path to dataset file (JSONL format)
            output_dir: Output directory for results
            max_samples: Maximum number of samples to process
            
        Returns:
            Processing summary
        """
        logger.info(f"Enhancing dataset questions from {dataset_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load dataset
        questions_data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    questions_data.append(item)
                    if max_samples and len(questions_data) >= max_samples:
                        break
                except:
                    continue
        
        logger.info(f"Loaded {len(questions_data)} questions from dataset")
        
        # Process questions
        enhanced_results = []
        failed_count = 0
        
        for i, item in enumerate(questions_data):
            try:
                logger.info(f"Processing {i+1}/{len(questions_data)}: {item.get('image_id', 'unknown')}")
                
                # Load image
                image_path = self._find_image_path(item['image_id'])
                if not image_path:
                    logger.warning(f"Image not found for {item['image_id']}")
                    failed_count += 1
                    continue
                
                image = Image.open(image_path).convert('RGB')
                question = item['question']
                
                # Enhance question
                enhanced_result = self.enhance_single_question(image, question)
                
                # Add metadata
                enhanced_result['dataset_metadata'] = {
                    'image_id': item['image_id'],
                    'original_answer': item.get('answer', ''),
                    'dataset_index': i
                }
                
                enhanced_results.append(enhanced_result)
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                failed_count += 1
        
        # Save results
        output_file = output_dir / 'enhanced_questions_dataset.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            'total_questions': len(questions_data),
            'successfully_enhanced': len(enhanced_results),
            'failed_processing': failed_count,
            'success_rate': len(enhanced_results) / len(questions_data),
            'average_quality_score': sum(r['quality_metrics']['overall_score'] 
                                       for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0,
            'output_file': str(output_file),
            'enhancement_stats': self.enhancement_stats.copy()
        }
        
        # Save summary
        summary_file = output_dir / 'enhancement_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset enhancement complete. Summary: {summary}")
        return summary
    
    def _find_image_path(self, image_id: str) -> Optional[str]:
        """Find image path for given image ID using config paths"""
        # Get image directories from config
        image_dirs = [
            self.config.get('data', {}).get('test_images', 'data/images/test'),
            self.config.get('data', {}).get('train_images', 'data/images/train'),
            self.config.get('data', {}).get('val_images', 'data/images/val')
        ]
        
        extensions = ['.jpg', '.jpeg', '.png']
        
        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                continue
                
            for ext in extensions:
                image_path = Path(image_dir) / f"{image_id}{ext}"
                if image_path.exists():
                    return str(image_path)
        
        return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_enhancement_statistics(self) -> Dict:
        """Get current enhancement statistics"""
        return self.enhancement_stats.copy()
    
    def reset_statistics(self):
        """Reset enhancement statistics"""
        self.enhancement_stats = {
            'total_processed': 0,
            'successful_reformulations': 0,
            'failed_reformulations': 0,
            'average_quality_score': 0.0
        }
        logger.info("Enhancement statistics reset")
