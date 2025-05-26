#!/usr/bin/env python
"""
🎯 PAPER EVALUATION SUITE - MedXplain-VQA
=============================================

Comprehensive quantitative evaluation suite for research paper preparation.
Provides BLEU, ROUGE, medical accuracy metrics, statistical analysis, and LaTeX table generation.

Author: MedXplain-VQA Project
Date: 2025-05-25
Purpose: Research paper quantitative evaluation
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core MedXplain imports - reuse existing infrastructure
from src.utils.config import Config
from src.utils.logger import setup_logger

# Import evaluation metrics libraries
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️ NLTK not available. Some metrics will be unavailable.")
    NLTK_AVAILABLE = False

try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    print("⚠️ ROUGE not available. Installing: pip install rouge")
    ROUGE_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.stats import ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ SciPy not available. Statistical tests will be limited.")
    SCIPY_AVAILABLE = False

# Import medxplain_vqa functions - REUSE EXISTING INFRASTRUCTURE
sys.path.append(os.path.join(os.path.dirname(__file__)))
try:
    from medxplain_vqa import (
        load_model, 
        initialize_explainable_components,
        load_test_samples,
        process_basic_vqa,
        process_explainable_vqa
    )
    MEDXPLAIN_AVAILABLE = True
    print("✅ MedXplain-VQA infrastructure loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import MedXplain-VQA components: {e}")
    MEDXPLAIN_AVAILABLE = False


class PaperEvaluationSuite:
    """
    🎯 COMPREHENSIVE PAPER EVALUATION SUITE
    
    Provides quantitative evaluation for MedXplain-VQA research paper including:
    - Multi-mode evaluation (basic, explainable, enhanced+bbox)
    - NLP metrics (BLEU-1,2,3,4 + ROUGE-L,1,2)
    - Medical accuracy assessment
    - Statistical analysis with confidence intervals
    - LaTeX table generation for paper
    """
    
    def __init__(self, config_path: str, model_path: str, output_dir: str = "data/paper_evaluation"):
        """
        Initialize evaluation suite with MedXplain-VQA infrastructure
        
        Args:
            config_path: Path to config.yaml
            model_path: Path to trained BLIP model
            output_dir: Directory for evaluation results
        """
        # Load configuration
        self.config = Config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            'paper_evaluation', 
            self.output_dir / 'logs',
            level='INFO'
        )
        
        self.logger.info("🚀 Initializing Paper Evaluation Suite")
        
        # Initialize MedXplain-VQA components
        if not MEDXPLAIN_AVAILABLE:
            raise RuntimeError("❌ MedXplain-VQA components not available")
            
        self.model_path = model_path
        self.blip_model = None
        self.components = None
        self._initialize_models()
        
        # Initialize metrics components
        self._initialize_metrics()
        
        # Evaluation modes
        self.evaluation_modes = {
            'basic': {'description': 'BLIP + Gemini', 'enable_cot': False, 'enable_bbox': False},
            'explainable': {'description': 'BLIP + Gemini + Query Reform + Grad-CAM', 'enable_cot': False, 'enable_bbox': False},
            'enhanced': {'description': 'Full MedXplain-VQA + Chain-of-Thought', 'enable_cot': True, 'enable_bbox': False},
            'enhanced_bbox': {'description': 'Full MedXplain-VQA + Bounding Boxes', 'enable_cot': True, 'enable_bbox': True}
        }
        
        self.logger.info(f"✅ Paper Evaluation Suite initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"🎯 Evaluation modes: {list(self.evaluation_modes.keys())}")
    
    def _initialize_models(self):
        """Initialize BLIP model and explainable components"""
        try:
            # Load BLIP model
            self.logger.info(f"Loading BLIP model from {self.model_path}")
            self.blip_model = load_model(self.config, self.model_path, self.logger)
            
            if self.blip_model is None:
                raise RuntimeError("Failed to load BLIP model")
            
            # Initialize explainable components (with bbox support)
            self.logger.info("Initializing explainable AI components")
            self.components = initialize_explainable_components(
                self.config, self.blip_model, enable_bbox=True, logger=self.logger
            )
            
            if self.components is None:
                raise RuntimeError("Failed to initialize explainable components")
                
            self.logger.info("✅ Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def _initialize_metrics(self):
        """Initialize metrics calculation components"""
        self.metrics_available = {
            'bleu': NLTK_AVAILABLE,
            'rouge': ROUGE_AVAILABLE, 
            'scipy': SCIPY_AVAILABLE
        }
        
        if ROUGE_AVAILABLE:
            self.rouge_evaluator = Rouge()
        
        if NLTK_AVAILABLE:
            self.smoothing = SmoothingFunction()
            
        self.logger.info(f"📊 Metrics availability: {self.metrics_available}")
    
    def load_stratified_samples(self, num_samples: int = 100, random_seed: int = 42) -> List[Dict]:
        """
        Load stratified samples for balanced evaluation
        
        Args:
            num_samples: Total number of samples to load
            random_seed: Random seed for reproducibility
            
        Returns:
            List of stratified samples with metadata
        """
        self.logger.info(f"📊 Loading {num_samples} stratified samples (seed: {random_seed})")
        
        try:
            # Load all available samples first
            all_samples = load_test_samples(self.config, num_samples=num_samples*3, random_seed=random_seed)
            
            if len(all_samples) < num_samples:
                self.logger.warning(f"⚠️ Only {len(all_samples)} samples available, requested {num_samples}")
                num_samples = len(all_samples)
            
            # Analyze questions for stratification
            question_types = self._categorize_questions([s['question'] for s in all_samples])
            
            # Stratified sampling by question type
            stratified_samples = self._stratified_sampling(all_samples, question_types, num_samples)
            
            self.logger.info(f"✅ Loaded {len(stratified_samples)} stratified samples")
            self._log_sample_distribution(stratified_samples, question_types)
            
            return stratified_samples
            
        except Exception as e:
            self.logger.error(f"❌ Error loading stratified samples: {e}")
            raise
    
    def _categorize_questions(self, questions: List[str]) -> List[str]:
        """Categorize questions by type for stratification"""
        categories = []
        
        for question in questions:
            q_lower = question.lower()
            
            if any(word in q_lower for word in ['what', 'describe', 'show']):
                categories.append('descriptive')
            elif any(word in q_lower for word in ['is', 'are', 'can you see', 'present']):
                categories.append('presence')  
            elif any(word in q_lower for word in ['diagnos', 'condition', 'disease']):
                categories.append('diagnostic')
            elif any(word in q_lower for word in ['compare', 'difference', 'versus']):
                categories.append('comparative')
            else:
                categories.append('other')
                
        return categories
    
    def _stratified_sampling(self, samples: List[Dict], categories: List[str], num_samples: int) -> List[Dict]:
        """Perform stratified sampling to ensure balanced question types"""
        # Group samples by category
        samples_by_category = defaultdict(list)
        for sample, category in zip(samples, categories):
            samples_by_category[category].append(sample)
        
        # Calculate samples per category
        category_counts = Counter(categories)
        stratified_samples = []
        
        for category, available_samples in samples_by_category.items():
            # Proportional sampling
            proportion = category_counts[category] / len(categories)
            target_count = max(1, int(num_samples * proportion))
            
            # Sample from this category
            if len(available_samples) >= target_count:
                selected = random.sample(available_samples, target_count)
            else:
                selected = available_samples
                
            stratified_samples.extend(selected)
        
        # Fill remaining slots randomly if needed
        while len(stratified_samples) < num_samples:
            remaining_samples = [s for s in samples if s not in stratified_samples]
            if not remaining_samples:
                break
            stratified_samples.append(random.choice(remaining_samples))
        
        return stratified_samples[:num_samples]
    
    def _log_sample_distribution(self, samples: List[Dict], categories: List[str]):
        """Log sample distribution for transparency"""
        category_dist = Counter(categories[:len(samples)])
        
        self.logger.info("📊 Sample distribution:")
        for category, count in category_dist.items():
            percentage = count / len(samples) * 100
            self.logger.info(f"  {category}: {count} samples ({percentage:.1f}%)")
    
    def run_comprehensive_evaluation(self, num_samples: int = 100, modes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        🎯 MAIN EVALUATION PIPELINE
        
        Args:
            num_samples: Number of samples to evaluate
            modes: List of modes to evaluate (default: all modes)
            
        Returns:
            Comprehensive evaluation results
        """
        if modes is None:
            modes = list(self.evaluation_modes.keys())
            
        self.logger.info(f"🚀 Starting comprehensive evaluation")
        self.logger.info(f"📊 Samples: {num_samples}, Modes: {modes}")
        
        # Step 1: Load stratified samples
        samples = self.load_stratified_samples(num_samples)
        
        # Step 2: Run evaluation for each mode
        all_results = {}
        
        for mode_name in modes:
            if mode_name not in self.evaluation_modes:
                self.logger.warning(f"⚠️ Unknown mode: {mode_name}, skipping")
                continue
                
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔬 Evaluating mode: {mode_name}")
            self.logger.info(f"{'='*60}")
            
            mode_config = self.evaluation_modes[mode_name]
            mode_results = self._evaluate_mode(samples, mode_name, mode_config)
            all_results[mode_name] = mode_results
        
        # Step 3: Calculate comparative metrics
        comparative_results = self._calculate_comparative_metrics(all_results)
        
        # Step 4: Statistical analysis
        statistical_results = self.statistical_analysis(all_results)
        
        # Step 5: Generate comprehensive report
        final_results = {
            'evaluation_config': {
                'num_samples': len(samples),
                'modes_evaluated': modes,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'mode_results': all_results,
            'comparative_analysis': comparative_results,
            'statistical_analysis': statistical_results,
            'samples_metadata': {
                'total_samples': len(samples),
                'sample_ids': [s['image_id'] for s in samples[:10]]  # First 10 for reference
            }
        }
        
        # Save results
        self._save_evaluation_results(final_results)
        
        self.logger.info(f"\n🎉 Comprehensive evaluation completed!")
        self.logger.info(f"📁 Results saved to: {self.output_dir}")
        
        return final_results
    
    def _evaluate_mode(self, samples: List[Dict], mode_name: str, mode_config: Dict) -> Dict[str, Any]:
        """Evaluate specific mode on all samples"""
        enable_cot = mode_config['enable_cot']
        enable_bbox = mode_config['enable_bbox']
        
        predictions = []
        ground_truths = []
        processing_times = []
        detailed_results = []
        
        total_samples = len(samples)
        successful_samples = 0
        
        for i, sample in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{total_samples}: {sample['image_id']}")
            
            try:
                import time
                start_time = time.time()
                
                # Process sample based on mode
                if mode_name == 'basic':
                    result = process_basic_vqa(
                        self.blip_model, 
                        self.components['gemini'], 
                        sample, 
                        self.logger
                    )
                else:
                    # Update components bbox setting for this evaluation
                    if enable_bbox != self.components.get('bbox_enabled', False):
                        # Re-initialize components with correct bbox setting
                        self.components = initialize_explainable_components(
                            self.config, self.blip_model, enable_bbox, self.logger
                        )
                    
                    result = process_explainable_vqa(
                        self.blip_model,
                        self.components, 
                        sample,
                        enable_cot,
                        self.logger
                    )
                
                processing_time = time.time() - start_time
                
                if result['success']:
                    predictions.append(result['unified_answer'])
                    ground_truths.append(sample['answer'])
                    processing_times.append(processing_time)
                    successful_samples += 1
                    
                    # Store detailed result for analysis
                    detailed_result = {
                        'sample_id': sample['image_id'],
                        'question': sample['question'],
                        'ground_truth': sample['answer'],
                        'prediction': result['unified_answer'],
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    # Add mode-specific metadata
                    if mode_name != 'basic':
                        detailed_result.update({
                            'reformulation_quality': result.get('reformulation_quality', 0),
                            'bbox_regions_count': len(result.get('bbox_regions', [])),
                        })
                        
                        if enable_cot and result.get('reasoning_result'):
                            reasoning = result['reasoning_result']
                            if reasoning['success']:
                                detailed_result['reasoning_confidence'] = reasoning['reasoning_chain']['overall_confidence']
                    
                    detailed_results.append(detailed_result)
                else:
                    self.logger.warning(f"⚠️ Sample {sample['image_id']} failed: {result.get('error_messages', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing sample {sample['image_id']}: {e}")
                continue
        
        # Calculate metrics
        metrics = self.calculate_nlp_metrics(predictions, ground_truths)
        
        # Add processing statistics
        metrics.update({
            'processing_stats': {
                'total_samples': total_samples,
                'successful_samples': successful_samples,
                'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
                'average_processing_time': np.mean(processing_times) if processing_times else 0,
                'std_processing_time': np.std(processing_times) if processing_times else 0
            },
            'detailed_results': detailed_results
        })
        
        self.logger.info(f"✅ Mode {mode_name} completed: {successful_samples}/{total_samples} samples successful")
        
        return metrics
    
    def calculate_nlp_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive NLP metrics
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary of calculated metrics
        """
        if not predictions or not ground_truths:
            self.logger.warning("⚠️ Empty predictions or ground truths for metrics calculation")
            return self._get_empty_metrics()
        
        if len(predictions) != len(ground_truths):
            self.logger.warning(f"⚠️ Mismatch in predictions ({len(predictions)}) and ground truths ({len(ground_truths)})")
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]
        
        metrics = {}
        
        # BLEU scores
        if self.metrics_available['bleu']:
            metrics.update(self._calculate_bleu_scores(predictions, ground_truths))
        
        # ROUGE scores  
        if self.metrics_available['rouge']:
            metrics.update(self._calculate_rouge_scores(predictions, ground_truths))
        
        # Medical accuracy metrics
        metrics.update(self._calculate_medical_accuracy(predictions, ground_truths))
        
        # Basic string metrics
        metrics.update(self._calculate_basic_metrics(predictions, ground_truths))
        
        self.logger.info(f"📊 Calculated metrics for {len(predictions)} samples")
        
        return metrics
    
    def _calculate_bleu_scores(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores"""
        try:
            # Tokenize
            pred_tokens = [pred.split() for pred in predictions]
            ref_tokens = [[ref.split()] for ref in ground_truths]  # List of list for corpus_bleu
            
            bleu_scores = {}
            
            # Calculate BLEU-1 to BLEU-4
            for n in range(1, 5):
                weights = tuple([1.0/n if i < n else 0.0 for i in range(4)])
                
                # Corpus-level BLEU
                corpus_bleu_score = corpus_bleu(
                    ref_tokens, pred_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
                
                # Average sentence-level BLEU 
                sentence_bleu_scores = []
                for pred_tok, ref_tok in zip(pred_tokens, ref_tokens):
                    score = sentence_bleu(
                        ref_tok, pred_tok,
                        weights=weights,
                        smoothing_function=self.smoothing.method1
                    )
                    sentence_bleu_scores.append(score)
                
                avg_sentence_bleu = np.mean(sentence_bleu_scores)
                
                bleu_scores[f'bleu_{n}'] = corpus_bleu_score
                bleu_scores[f'bleu_{n}_avg'] = avg_sentence_bleu
            
            return bleu_scores
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating BLEU scores: {e}")
            return {f'bleu_{n}': 0.0 for n in range(1, 5)}
    
    def _calculate_rouge_scores(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, ROUGE-L scores"""
        try:
            # Calculate ROUGE scores
            rouge_scores = self.rouge_evaluator.get_scores(predictions, ground_truths, avg=True)
            
            return {
                'rouge_1_f': rouge_scores['rouge-1']['f'],
                'rouge_1_p': rouge_scores['rouge-1']['p'], 
                'rouge_1_r': rouge_scores['rouge-1']['r'],
                'rouge_2_f': rouge_scores['rouge-2']['f'],
                'rouge_2_p': rouge_scores['rouge-2']['p'],
                'rouge_2_r': rouge_scores['rouge-2']['r'],
                'rouge_l_f': rouge_scores['rouge-l']['f'],
                'rouge_l_p': rouge_scores['rouge-l']['p'],
                'rouge_l_r': rouge_scores['rouge-l']['r']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ROUGE scores: {e}")
            return {f'rouge_{metric}': 0.0 for metric in ['1_f', '1_p', '1_r', '2_f', '2_p', '2_r', 'l_f', 'l_p', 'l_r']}
    
    def _calculate_medical_accuracy(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate medical domain-specific accuracy metrics"""
        try:
            # Exact match accuracy
            exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) 
                              if pred.strip().lower() == gt.strip().lower())
            exact_match_accuracy = exact_matches / len(predictions)
            
            # Substring match accuracy (partial credit)
            substring_matches = 0
            for pred, gt in zip(predictions, ground_truths):
                pred_words = set(pred.lower().split())
                gt_words = set(gt.lower().split())
                
                if pred_words & gt_words:  # Non-empty intersection
                    substring_matches += 1
            
            substring_accuracy = substring_matches / len(predictions)
            
            # Medical keyword accuracy
            medical_keywords = {
                'pathology': ['cancer', 'tumor', 'malignant', 'benign', 'carcinoma', 'adenoma', 'melanoma'],
                'anatomy': ['tissue', 'cell', 'organ', 'epithelial', 'gland', 'vessel'],
                'diagnosis': ['normal', 'abnormal', 'inflammation', 'infection', 'disease']
            }
            
            keyword_accuracy = 0
            for pred, gt in zip(predictions, ground_truths):
                pred_lower = pred.lower()
                gt_lower = gt.lower()
                
                # Check if prediction contains relevant medical keywords from ground truth
                gt_keywords = []
                for category, keywords in medical_keywords.items():
                    gt_keywords.extend([kw for kw in keywords if kw in gt_lower])
                
                if gt_keywords:
                    matches = sum(1 for kw in gt_keywords if kw in pred_lower)
                    keyword_accuracy += matches / len(gt_keywords)
                else:
                    keyword_accuracy += 1  # Full credit if no keywords expected
            
            keyword_accuracy /= len(predictions)
            
            return {
                'exact_match_accuracy': exact_match_accuracy,
                'substring_accuracy': substring_accuracy,
                'medical_keyword_accuracy': keyword_accuracy
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating medical accuracy: {e}")
            return {'exact_match_accuracy': 0.0, 'substring_accuracy': 0.0, 'medical_keyword_accuracy': 0.0}
    
    def _calculate_basic_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate basic string-based metrics"""
        try:
            # Average answer length
            avg_pred_length = np.mean([len(pred.split()) for pred in predictions])
            avg_gt_length = np.mean([len(gt.split()) for gt in ground_truths])
            
            # Length ratio
            length_ratio = avg_pred_length / avg_gt_length if avg_gt_length > 0 else 0
            
            return {
                'avg_prediction_length': avg_pred_length,
                'avg_ground_truth_length': avg_gt_length,
                'length_ratio': length_ratio
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating basic metrics: {e}")
            return {'avg_prediction_length': 0.0, 'avg_ground_truth_length': 0.0, 'length_ratio': 0.0}
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure"""
        return {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0,
            'exact_match_accuracy': 0.0, 'substring_accuracy': 0.0,
            'medical_keyword_accuracy': 0.0
        }
    
    def _calculate_comparative_metrics(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparative metrics across modes"""
        if len(all_results) < 2:
            return {}
        
        comparative_results = {
            'metric_comparisons': {},
            'improvement_analysis': {},
            'ranking_analysis': {}
        }
        
        # Extract metrics for comparison
        mode_metrics = {}
        for mode_name, results in all_results.items():
            mode_metrics[mode_name] = {k: v for k, v in results.items() 
                                     if isinstance(v, (int, float)) and k != 'processing_stats'}
        
        # Calculate improvements relative to basic mode
        if 'basic' in mode_metrics:
            baseline_metrics = mode_metrics['basic']
            
            for mode_name, metrics in mode_metrics.items():
                if mode_name == 'basic':
                    continue
                    
                improvements = {}
                for metric_name, value in metrics.items():
                    if metric_name in baseline_metrics:
                        baseline_value = baseline_metrics[metric_name]
                        if baseline_value > 0:
                            improvement = (value - baseline_value) / baseline_value * 100
                            improvements[metric_name] = improvement
                
                comparative_results['improvement_analysis'][mode_name] = improvements
        
        # Rank modes by key metrics
        key_metrics = ['bleu_4', 'rouge_l_f', 'exact_match_accuracy']
        rankings = {}
        
        for metric in key_metrics:
            metric_values = []
            for mode_name, metrics in mode_metrics.items():
                if metric in metrics:
                    metric_values.append((mode_name, metrics[metric]))
            
            # Sort by metric value (descending)
            metric_values.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [mode for mode, value in metric_values]
        
        comparative_results['ranking_analysis'] = rankings
        
        return comparative_results
    
    def statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis with confidence intervals and significance testing
        
        Args:
            results: Results from all evaluation modes
            
        Returns:
            Statistical analysis results
        """
        self.logger.info("📊 Performing statistical analysis")
        
        if not self.metrics_available['scipy']:
            self.logger.warning("⚠️ SciPy not available, statistical tests limited")
            return self._basic_statistical_analysis(results)
        
        statistical_results = {
            'descriptive_statistics': {},
            'confidence_intervals': {},
            'significance_tests': {},
            'effect_sizes': {}
        }
        
        # Extract detailed results for statistical analysis
        mode_data = {}
        for mode_name, mode_results in results.items():
            if 'detailed_results' in mode_results:
                mode_data[mode_name] = mode_results['detailed_results']
        
        if len(mode_data) < 2:
            self.logger.warning("⚠️ Insufficient modes for statistical comparison")
            return statistical_results
        
        # Calculate descriptive statistics
        statistical_results['descriptive_statistics'] = self._calculate_descriptive_stats(results)
        
        # Calculate confidence intervals
        statistical_results['confidence_intervals'] = self._calculate_confidence_intervals(mode_data)
        
        # Perform significance tests
        statistical_results['significance_tests'] = self._perform_significance_tests(mode_data)
        
        # Calculate effect sizes
        statistical_results['effect_sizes'] = self._calculate_effect_sizes(mode_data)
        
        self.logger.info("✅ Statistical analysis completed")
        
        return statistical_results
    
    def _calculate_descriptive_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate descriptive statistics for each mode"""
        descriptive_stats = {}
        
        for mode_name, mode_results in results.items():
            stats = {}
            
            # Extract numeric metrics
            for key, value in mode_results.items():
                if isinstance(value, (int, float)) and key != 'processing_stats':
                    stats[key] = {
                        'value': value,
                        'type': 'single_value'
                    }
            
            # Processing time statistics
            if 'processing_stats' in mode_results:
                proc_stats = mode_results['processing_stats']
                stats['processing_time'] = {
                    'mean': proc_stats.get('average_processing_time', 0),
                    'std': proc_stats.get('std_processing_time', 0)
                }
            
            descriptive_stats[mode_name] = stats
        
        return descriptive_stats
    
    def _calculate_confidence_intervals(self, mode_data: Dict[str, List[Dict]], confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate confidence intervals for key metrics"""
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for mode_name, detailed_results in mode_data.items():
            if not detailed_results:
                continue
                
            mode_cis = {}
            
            # Extract processing times for CI calculation
            processing_times = [r['processing_time'] for r in detailed_results if 'processing_time' in r]
            
            if processing_times:
                mean_time = np.mean(processing_times)
                std_time = np.std(processing_times, ddof=1)
                n = len(processing_times)
                
                # t-distribution critical value
                t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
                margin_error = t_critical * (std_time / np.sqrt(n))
                
                mode_cis['processing_time'] = {
                    'mean': mean_time,
                    'lower_bound': mean_time - margin_error,
                    'upper_bound': mean_time + margin_error,
                    'margin_error': margin_error,
                    'confidence_level': confidence_level
                }
            
            confidence_intervals[mode_name] = mode_cis
        
        return confidence_intervals
    
    def _perform_significance_tests(self, mode_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform statistical significance tests between modes"""
        significance_tests = {}
        mode_names = list(mode_data.keys())
        
        # Pairwise comparisons
        for i, mode1 in enumerate(mode_names):
            for mode2 in mode_names[i+1:]:
                test_key = f"{mode1}_vs_{mode2}"
                
                # Get processing times for both modes
                times1 = [r['processing_time'] for r in mode_data[mode1] if 'processing_time' in r]
                times2 = [r['processing_time'] for r in mode_data[mode2] if 'processing_time' in r]
                
                if len(times1) > 1 and len(times2) > 1:
                    # Perform t-test
                    t_stat, p_value = ttest_ind(times1, times2)
                    
                    # Perform Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = mannwhitneyu(times1, times2, alternative='two-sided')
                    
                    significance_tests[test_key] = {
                        'ttest': {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        },
                        'mannwhitney': {
                            'u_statistic': u_stat,
                            'p_value': u_p_value,
                            'significant': u_p_value < 0.05
                        },
                        'sample_sizes': {
                            mode1: len(times1),
                            mode2: len(times2)
                        }
                    }
        
        return significance_tests
    
    def _calculate_effect_sizes(self, mode_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate effect sizes (Cohen's d) for pairwise comparisons"""
        effect_sizes = {}
        mode_names = list(mode_data.keys())
        
        for i, mode1 in enumerate(mode_names):
            for mode2 in mode_names[i+1:]:
                test_key = f"{mode1}_vs_{mode2}"
                
                times1 = [r['processing_time'] for r in mode_data[mode1] if 'processing_time' in r]
                times2 = [r['processing_time'] for r in mode_data[mode2] if 'processing_time' in r]
                
                if len(times1) > 1 and len(times2) > 1:
                    # Calculate Cohen's d
                    mean1, mean2 = np.mean(times1), np.mean(times2)
                    std1, std2 = np.std(times1, ddof=1), np.std(times2, ddof=1)
                    n1, n2 = len(times1), len(times2)
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    # Effect size interpretation
                    if abs(cohens_d) < 0.2:
                        interpretation = 'negligible'
                    elif abs(cohens_d) < 0.5:
                        interpretation = 'small'
                    elif abs(cohens_d) < 0.8:
                        interpretation = 'medium'
                    else:
                        interpretation = 'large'
                    
                    effect_sizes[test_key] = {
                        'cohens_d': cohens_d,
                        'interpretation': interpretation,
                        'means': {mode1: mean1, mode2: mean2},
                        'std_devs': {mode1: std1, mode2: std2}
                    }
        
        return effect_sizes
    
    def _basic_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Basic statistical analysis when SciPy unavailable"""
        basic_stats = {}
        
        for mode_name, mode_results in results.items():
            mode_stats = {}
            
            if 'processing_stats' in mode_results:
                proc_stats = mode_results['processing_stats']
                mode_stats['processing_time'] = {
                    'mean': proc_stats.get('average_processing_time', 0),
                    'std': proc_stats.get('std_processing_time', 0),
                    'success_rate': proc_stats.get('success_rate', 0)
                }
            
            basic_stats[mode_name] = mode_stats
        
        return {'basic_statistics': basic_stats}
    
    def export_paper_tables(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate LaTeX tables ready for paper inclusion
        
        Args:
            results: Comprehensive evaluation results
            
        Returns:
            Dictionary with LaTeX table files
        """
        self.logger.info("📝 Generating LaTeX tables for paper")
        
        latex_files = {}
        
        # Table 1: Quantitative Performance Comparison
        table1_file = self._generate_performance_table(results)
        if table1_file:
            latex_files['performance_table'] = table1_file
        
        # Table 2: Statistical Analysis Summary
        table2_file = self._generate_statistical_table(results)
        if table2_file:
            latex_files['statistical_table'] = table2_file
        
        # Table 3: Processing Efficiency Comparison
        table3_file = self._generate_efficiency_table(results)
        if table3_file:
            latex_files['efficiency_table'] = table3_file
        
        self.logger.info(f"✅ Generated {len(latex_files)} LaTeX tables")
        
        return latex_files
    
    def _generate_performance_table(self, results: Dict[str, Any]) -> Optional[str]:
        """Generate main performance comparison table"""
        try:
            table_file = self.output_dir / "performance_comparison_table.tex"
            
            with open(table_file, 'w') as f:
                f.write("% MedXplain-VQA Performance Comparison Table\n")
                f.write("% Generated automatically by Paper Evaluation Suite\n\n")
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{Quantitative Performance Comparison of MedXplain-VQA Modes}\n")
                f.write("\\label{tab:performance_comparison}\n")
                f.write("\\begin{tabular}{lccccc}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Mode} & \\textbf{BLEU-4} & \\textbf{ROUGE-L} & \\textbf{Exact Match} & \\textbf{Medical Acc.} & \\textbf{Processing (s)} \\\\\n")
                f.write("\\midrule\n")
                
                # Get mode results
                mode_results = results.get('mode_results', {})
                
                for mode_name, mode_data in mode_results.items():
                    # Format mode name
                    display_name = mode_name.replace('_', ' ').title()
                    
                    # Extract metrics with fallbacks
                    bleu4 = mode_data.get('bleu_4', 0.0)
                    rouge_l = mode_data.get('rouge_l_f', 0.0)
                    exact_match = mode_data.get('exact_match_accuracy', 0.0)
                    medical_acc = mode_data.get('medical_keyword_accuracy', 0.0)
                    
                    proc_stats = mode_data.get('processing_stats', {})
                    avg_time = proc_stats.get('average_processing_time', 0.0)
                    
                    f.write(f"{display_name} & {bleu4:.3f} & {rouge_l:.3f} & {exact_match:.3f} & {medical_acc:.3f} & {avg_time:.1f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            self.logger.info(f"✅ Performance table saved: {table_file}")
            return str(table_file)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance table: {e}")
            return None
    
    def _generate_statistical_table(self, results: Dict[str, Any]) -> Optional[str]:
        """Generate statistical analysis summary table"""
        try:
            table_file = self.output_dir / "statistical_analysis_table.tex"
            
            statistical_analysis = results.get('statistical_analysis', {})
            significance_tests = statistical_analysis.get('significance_tests', {})
            
            if not significance_tests:
                self.logger.warning("⚠️ No significance tests available for table generation")
                return None
            
            with open(table_file, 'w') as f:
                f.write("% MedXplain-VQA Statistical Analysis Table\n")
                f.write("% Generated automatically by Paper Evaluation Suite\n\n")
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{Statistical Significance Analysis Between MedXplain-VQA Modes}\n")
                f.write("\\label{tab:statistical_analysis}\n")
                f.write("\\begin{tabular}{lccc}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Comparison} & \\textbf{t-test p-value} & \\textbf{Mann-Whitney p-value} & \\textbf{Significant} \\\\\n")
                f.write("\\midrule\n")
                
                for comparison, test_results in significance_tests.items():
                    # Format comparison name
                    comparison_name = comparison.replace('_vs_', ' vs ').replace('_', ' ').title()
                    
                    t_p = test_results.get('ttest', {}).get('p_value', 1.0)
                    u_p = test_results.get('mannwhitney', {}).get('p_value', 1.0)
                    
                    # Determine overall significance
                    significant = t_p < 0.05 or u_p < 0.05
                    sig_symbol = "\\textbf{Yes}" if significant else "No"
                    
                    f.write(f"{comparison_name} & {t_p:.4f} & {u_p:.4f} & {sig_symbol} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            self.logger.info(f"✅ Statistical table saved: {table_file}")
            return str(table_file)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating statistical table: {e}")
            return None
    
    def _generate_efficiency_table(self, results: Dict[str, Any]) -> Optional[str]:
        """Generate processing efficiency comparison table"""
        try:
            table_file = self.output_dir / "efficiency_comparison_table.tex"
            
            with open(table_file, 'w') as f:
                f.write("% MedXplain-VQA Efficiency Comparison Table\n")
                f.write("% Generated automatically by Paper Evaluation Suite\n\n")
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{Processing Efficiency Comparison of MedXplain-VQA Modes}\n")
                f.write("\\label{tab:efficiency_comparison}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Mode} & \\textbf{Avg Time (s)} & \\textbf{Std Time (s)} & \\textbf{Success Rate} & \\textbf{Samples/min} \\\\\n")
                f.write("\\midrule\n")
                
                mode_results = results.get('mode_results', {})
                
                for mode_name, mode_data in mode_results.items():
                    display_name = mode_name.replace('_', ' ').title()
                    
                    proc_stats = mode_data.get('processing_stats', {})
                    avg_time = proc_stats.get('average_processing_time', 0.0)
                    std_time = proc_stats.get('std_processing_time', 0.0)
                    success_rate = proc_stats.get('success_rate', 0.0)
                    
                    # Calculate samples per minute
                    samples_per_min = 60.0 / avg_time if avg_time > 0 else 0.0
                    
                    f.write(f"{display_name} & {avg_time:.1f} & {std_time:.1f} & {success_rate:.3f} & {samples_per_min:.1f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            self.logger.info(f"✅ Efficiency table saved: {table_file}")
            return str(table_file)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating efficiency table: {e}")
            return None
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results"""
        try:
            # Save main results as JSON
            results_file = self.output_dir / "comprehensive_evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate LaTeX tables
            latex_files = self.export_paper_tables(results)
            
            # Save summary report
            self._generate_summary_report(results)
            
            self.logger.info(f"✅ All evaluation results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving evaluation results: {e}")
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report"""
        try:
            report_file = self.output_dir / "evaluation_summary_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("🎯 MEDXPLAIN-VQA COMPREHENSIVE EVALUATION SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                # Evaluation overview
                config = results.get('evaluation_config', {})
                f.write(f"📊 EVALUATION OVERVIEW\n")
                f.write(f"  Samples evaluated: {config.get('num_samples', 'Unknown')}\n")
                f.write(f"  Modes evaluated: {', '.join(config.get('modes_evaluated', []))}\n")
                f.write(f"  Timestamp: {config.get('timestamp', 'Unknown')}\n\n")
                
                # Performance summary
                f.write(f"🏆 PERFORMANCE SUMMARY\n")
                mode_results = results.get('mode_results', {})
                
                for mode_name, mode_data in mode_results.items():
                    f.write(f"\n  {mode_name.upper()}:\n")
                    f.write(f"    BLEU-4: {mode_data.get('bleu_4', 0.0):.3f}\n")
                    f.write(f"    ROUGE-L: {mode_data.get('rouge_l_f', 0.0):.3f}\n")
                    f.write(f"    Exact Match: {mode_data.get('exact_match_accuracy', 0.0):.3f}\n")
                    f.write(f"    Medical Accuracy: {mode_data.get('medical_keyword_accuracy', 0.0):.3f}\n")
                    
                    proc_stats = mode_data.get('processing_stats', {})
                    f.write(f"    Processing Time: {proc_stats.get('average_processing_time', 0.0):.1f}s\n")
                    f.write(f"    Success Rate: {proc_stats.get('success_rate', 0.0):.3f}\n")
                
                # Statistical significance
                statistical_analysis = results.get('statistical_analysis', {})
                significance_tests = statistical_analysis.get('significance_tests', {})
                
                if significance_tests:
                    f.write(f"\n📈 STATISTICAL SIGNIFICANCE\n")
                    for comparison, test_results in significance_tests.items():
                        t_significant = test_results.get('ttest', {}).get('significant', False)
                        u_significant = test_results.get('mannwhitney', {}).get('significant', False)
                        overall_significant = t_significant or u_significant
                        
                        status = "SIGNIFICANT" if overall_significant else "Not significant"
                        f.write(f"  {comparison}: {status}\n")
                
                f.write(f"\n✅ Evaluation completed successfully!\n")
                f.write(f"📁 Detailed results available in: {self.output_dir}\n")
            
            self.logger.info(f"✅ Summary report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary report: {e}")


def test_paper_evaluation_suite():
    """
    🧪 TEST FUNCTION - Paper Evaluation Suite
    """
    print("🧪 Testing Paper Evaluation Suite")
    
    # Test configuration
    config_path = "configs/config.yaml"
    model_path = "checkpoints/blip/checkpoints/best_hf_model"
    test_output_dir = "data/paper_evaluation_test"
    
    try:
        # Test 1: Initialization
        print("\n1️⃣ Testing initialization...")
        evaluation_suite = PaperEvaluationSuite(
            config_path=config_path,
            model_path=model_path,
            output_dir=test_output_dir
        )
        print("✅ Initialization successful")
        
        # Test 2: Load stratified samples
        print("\n2️⃣ Testing stratified sample loading...")
        samples = evaluation_suite.load_stratified_samples(num_samples=5)
        print(f"✅ Loaded {len(samples)} stratified samples")
        
        # Test 3: Calculate NLP metrics (synthetic data)
        print("\n3️⃣ Testing NLP metrics calculation...")
        test_predictions = [
            "This image shows melanoma cells",
            "Normal tissue with no abnormalities", 
            "Inflammatory infiltrate present"
        ]
        test_ground_truths = [
            "Melanoma is visible in the tissue",
            "Normal healthy tissue",
            "Inflammation can be observed"
        ]
        
        metrics = evaluation_suite.calculate_nlp_metrics(test_predictions, test_ground_truths)
        print(f"✅ Calculated {len(metrics)} metrics")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.3f}")
        
        # Test 4: Mini evaluation (1 sample per mode)
        print("\n4️⃣ Testing mini evaluation...")
        mini_results = evaluation_suite.run_comprehensive_evaluation(
            num_samples=2,  # Very small for testing
            modes=['basic', 'explainable']  # Test subset of modes
        )
        print("✅ Mini evaluation completed")
        
        # Test 5: LaTeX table generation
        print("\n5️⃣ Testing LaTeX table generation...")
        latex_files = evaluation_suite.export_paper_tables(mini_results)
        print(f"✅ Generated {len(latex_files)} LaTeX tables")
        for table_name, file_path in latex_files.items():
            print(f"  {table_name}: {file_path}")
        
        print(f"\n🎉 All tests passed! Results in: {test_output_dir}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='🎯 Paper Evaluation Suite - MedXplain-VQA')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', help='BLIP model path')
    parser.add_argument('--output-dir', type=str, default='data/paper_evaluation', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--modes', nargs='+', default=None, help='Modes to evaluate (default: all)')
    parser.add_argument('--test', action='store_true', help='Run test function instead')
    
    args = parser.parse_args()
    
    if args.test:
        test_paper_evaluation_suite()
        return
    
    try:
        # Initialize evaluation suite
        evaluation_suite = PaperEvaluationSuite(
            config_path=args.config,
            model_path=args.model_path,
            output_dir=args.output_dir
        )
        
        # Run comprehensive evaluation
        results = evaluation_suite.run_comprehensive_evaluation(
            num_samples=args.num_samples,
            modes=args.modes
        )
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print key results
        mode_results = results.get('mode_results', {})
        print(f"\n📊 KEY RESULTS:")
        for mode_name, mode_data in mode_results.items():
            print(f"\n  {mode_name.upper()}:")
            print(f"    BLEU-4: {mode_data.get('bleu_4', 0.0):.3f}")
            print(f"    ROUGE-L: {mode_data.get('rouge_l_f', 0.0):.3f}")
            print(f"    Exact Match: {mode_data.get('exact_match_accuracy', 0.0):.3f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
