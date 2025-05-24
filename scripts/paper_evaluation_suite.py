#!/usr/bin/env python
"""
Paper Evaluation Suite for MedXplain-VQA
Comprehensive metrics collection for research publication
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import MedXplain components
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration
from src.explainability.reasoning.query_reformulator import QueryReformulator
from src.explainability.reasoning.visual_context_extractor import VisualContextExtractor
from src.explainability.enhanced_grad_cam import EnhancedGradCAM
from src.explainability.rationale.chain_of_thought import ChainOfThoughtGenerator

# NLTK imports for evaluation metrics
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("Warning: NLTK not available. Some metrics will be skipped.")

# ROUGE score implementation
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. Install with: pip install rouge-score")

class PaperEvaluationSuite:
    """Comprehensive evaluation suite for paper results"""
    
    def __init__(self, config_path: str, output_dir: str = "paper_results"):
        """Initialize evaluation suite"""
        self.config = Config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = setup_logger("paper_evaluation", str(self.output_dir), logging.INFO)
        
        # Initialize metrics
        self.metrics = {
            'bleu_scores': [],
            'rouge_scores': [],
            'meteor_scores': [],
            'clinical_accuracy': [],
            'reasoning_confidence': [],
            'processing_times': [],
            'attention_quality': [],
            'bbox_accuracy': [],
            'error_analysis': {}
        }
        
        # BLEU smoother
        self.bleu_smoother = SmoothingFunction().method1
        
        # ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.logger.info("Paper Evaluation Suite initialized")
    
    def load_test_samples(self, num_samples: int = 100, stratified: bool = True) -> List[Dict]:
        """Load test samples for evaluation"""
        self.logger.info(f"Loading {num_samples} test samples (stratified: {stratified})")
        
        # Load PathVQA test data
        test_questions_file = self.config['data']['test_questions']
        test_images_dir = self.config['data']['test_images']
        
        samples = []
        with open(test_questions_file, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        
        if stratified:
            # Stratify by question type and pathology
            stratified_samples = self._stratify_samples(all_data, num_samples)
            samples = stratified_samples
        else:
            # Random sampling
            np.random.seed(42)
            indices = np.random.choice(len(all_data), min(num_samples, len(all_data)), replace=False)
            samples = [all_data[i] for i in indices]
        
        self.logger.info(f"Loaded {len(samples)} samples for evaluation")
        return samples
    
    def _stratify_samples(self, all_data: List[Dict], num_samples: int) -> List[Dict]:
        """Stratify samples by question type and pathology"""
        
        # Categorize by question type
        question_categories = {
            'descriptive': [],
            'diagnostic': [],
            'presence': [],
            'comparison': [],
            'other': []
        }
        
        for item in all_data:
            question = item['question'].lower()
            if any(word in question for word in ['what', 'describe', 'show', 'see']):
                question_categories['descriptive'].append(item)
            elif any(word in question for word in ['diagnos', 'disease', 'condition']):
                question_categories['diagnostic'].append(item)
            elif any(word in question for word in ['is', 'are', 'present', 'visible']):
                question_categories['presence'].append(item)
            elif any(word in question for word in ['compare', 'difference', 'similar']):
                question_categories['comparison'].append(item)
            else:
                question_categories['other'].append(item)
        
        # Sample proportionally
        stratified = []
        samples_per_category = num_samples // len(question_categories)
        
        np.random.seed(42)
        for category, items in question_categories.items():
            if items:
                n_sample = min(samples_per_category, len(items))
                sampled = np.random.choice(len(items), n_sample, replace=False)
                stratified.extend([items[i] for i in sampled])
        
        # Fill remaining slots
        remaining = num_samples - len(stratified)
        if remaining > 0:
            remaining_items = [item for item in all_data if item not in stratified]
            if remaining_items:
                additional = np.random.choice(len(remaining_items), min(remaining, len(remaining_items)), replace=False)
                stratified.extend([remaining_items[i] for i in additional])
        
        return stratified[:num_samples]
    
    def run_comprehensive_evaluation(self, samples: List[Dict], modes: List[str] = None) -> Dict:
        """Run comprehensive evaluation on samples"""
        if modes is None:
            modes = ['basic', 'explainable', 'enhanced', 'enhanced_bbox']
        
        self.logger.info(f"Starting comprehensive evaluation on {len(samples)} samples")
        self.logger.info(f"Evaluation modes: {modes}")
        
        # Initialize components for each mode
        components = self._initialize_components()
        
        results = {}
        for mode in modes:
            self.logger.info(f"Evaluating mode: {mode}")
            mode_results = self._evaluate_mode(samples, mode, components)
            results[mode] = mode_results
            
            # Save intermediate results
            self._save_intermediate_results(mode, mode_results)
        
        # Comprehensive analysis
        comparative_analysis = self._comparative_analysis(results)
        results['comparative_analysis'] = comparative_analysis
        
        # Generate paper tables and figures
        self._generate_paper_outputs(results)
        
        return results
    
    def _initialize_components(self) -> Dict:
        """Initialize all MedXplain components"""
        self.logger.info("Initializing MedXplain components")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        components = {
            'blip_model': BLIP2VQA(self.config, train_mode=False).to(device),
            'gemini': GeminiIntegration(self.config),
            'query_reformulator': QueryReformulator(self.config),
            'visual_context': VisualContextExtractor(self.config),
            'enhanced_gradcam': EnhancedGradCAM(
                components['blip_model'] if 'blip_model' in locals() else None,
                bbox_config=self.config.get('explainability', {}).get('bounding_boxes', {})
            ),
            'chain_of_thought': ChainOfThoughtGenerator(
                components['gemini'] if 'gemini' in locals() else None, 
                self.config
            )
        }
        
        # Fix initialization order
        components['enhanced_gradcam'] = EnhancedGradCAM(
            components['blip_model'],
            bbox_config=self.config.get('explainability', {}).get('bounding_boxes', {})
        )
        components['chain_of_thought'] = ChainOfThoughtGenerator(
            components['gemini'], 
            self.config
        )
        
        self.logger.info("All components initialized successfully")
        return components
    
    def _evaluate_mode(self, samples: List[Dict], mode: str, components: Dict) -> Dict:
        """Evaluate specific mode on samples"""
        results = {
            'predictions': [],
            'metrics': {},
            'processing_times': [],
            'errors': []
        }
        
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {mode}")):
            try:
                start_time = datetime.now()
                
                # Process sample based on mode
                prediction = self._process_sample(sample, mode, components)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Calculate metrics for this sample
                sample_metrics = self._calculate_sample_metrics(sample, prediction)
                
                results['predictions'].append({
                    'sample_id': sample.get('image_id', f'sample_{i}'),
                    'prediction': prediction,
                    'ground_truth': sample['answer'],
                    'question': sample['question'],
                    'metrics': sample_metrics,
                    'processing_time': processing_time
                })
                
                results['processing_times'].append(processing_time)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {i} in mode {mode}: {e}")
                results['errors'].append({
                    'sample_id': sample.get('image_id', f'sample_{i}'),
                    'error': str(e)
                })
        
        # Aggregate metrics
        results['metrics'] = self._aggregate_metrics(results['predictions'])
        
        return results
    
    def _process_sample(self, sample: Dict, mode: str, components: Dict) -> Dict:
        """Process single sample based on mode"""
        from PIL import Image
        
        # Load image
        image_path = os.path.join(self.config['data']['test_images'], f"{sample['image_id']}.jpg")
        if not os.path.exists(image_path):
            # Try alternative extensions
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(self.config['data']['test_images'], f"{sample['image_id']}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        image = Image.open(image_path).convert('RGB')
        question = sample['question']
        
        if mode == 'basic':
            # Basic BLIP only
            blip_answer = components['blip_model'].predict(image, question)
            return {
                'answer': blip_answer,
                'processing_steps': ['blip_inference'],
                'confidence': None,
                'attention_data': None
            }
        
        elif mode == 'explainable':
            # BLIP + Query Reformulation + Grad-CAM
            blip_answer = components['blip_model'].predict(image, question)
            
            # Query reformulation
            reformulated = components['query_reformulator'].reformulate_question(image, question)
            
            # Basic Grad-CAM
            gradcam_result = components['enhanced_gradcam'].analyze_image_with_question(image, question)
            
            # Gemini enhancement
            unified_answer = components['gemini'].generate_unified_answer(
                image, question, blip_answer, 
                heatmap=gradcam_result.get('heatmap')
            )
            
            return {
                'answer': unified_answer,
                'blip_answer': blip_answer,
                'reformulated_question': reformulated['reformulated_question'],
                'reformulation_quality': reformulated['quality_score'],
                'processing_steps': ['blip_inference', 'query_reformulation', 'gradcam', 'gemini_enhancement'],
                'confidence': None,
                'attention_data': gradcam_result
            }
        
        elif mode == 'enhanced':
            # Full pipeline without bounding boxes
            blip_answer = components['blip_model'].predict(image, question)
            
            reformulated = components['query_reformulator'].reformulate_question(image, question)
            
            visual_context = components['visual_context'].extract_visual_context(image, question)
            
            gradcam_result = components['enhanced_gradcam'].analyze_image_with_question(image, question)
            
            # Chain-of-Thought reasoning
            reasoning_result = components['chain_of_thought'].generate_reasoning_chain(
                image, reformulated['reformulated_question'], blip_answer, 
                visual_context, grad_cam_data=gradcam_result
            )
            
            unified_answer = components['gemini'].generate_unified_answer(
                image, question, blip_answer,
                heatmap=gradcam_result.get('heatmap')
            )
            
            return {
                'answer': unified_answer,
                'blip_answer': blip_answer,
                'reformulated_question': reformulated['reformulated_question'],
                'reformulation_quality': reformulated['quality_score'],
                'reasoning_result': reasoning_result,
                'processing_steps': ['blip_inference', 'query_reformulation', 'visual_context', 'gradcam', 'chain_of_thought', 'gemini_enhancement'],
                'confidence': reasoning_result.get('reasoning_chain', {}).get('overall_confidence'),
                'attention_data': gradcam_result
            }
        
        elif mode == 'enhanced_bbox':
            # Full pipeline with bounding boxes
            blip_answer = components['blip_model'].predict(image, question)
            
            reformulated = components['query_reformulator'].reformulate_question(image, question)
            
            visual_context = components['visual_context'].extract_visual_context(image, question)
            
            # Enhanced Grad-CAM with bounding boxes
            gradcam_result = components['enhanced_gradcam'].analyze_image_with_question(image, question)
            
            reasoning_result = components['chain_of_thought'].generate_reasoning_chain(
                image, reformulated['reformulated_question'], blip_answer, 
                visual_context, grad_cam_data=gradcam_result
            )
            
            unified_answer = components['gemini'].generate_unified_answer(
                image, question, blip_answer,
                heatmap=gradcam_result.get('heatmap')
            )
            
            return {
                'answer': unified_answer,
                'blip_answer': blip_answer,
                'reformulated_question': reformulated['reformulated_question'],
                'reformulation_quality': reformulated['quality_score'],
                'reasoning_result': reasoning_result,
                'processing_steps': ['blip_inference', 'query_reformulation', 'visual_context', 'enhanced_gradcam_bbox', 'chain_of_thought', 'gemini_enhancement'],
                'confidence': reasoning_result.get('reasoning_chain', {}).get('overall_confidence'),
                'attention_data': gradcam_result,
                'bounding_boxes': gradcam_result.get('regions', [])
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _calculate_sample_metrics(self, sample: Dict, prediction: Dict) -> Dict:
        """Calculate metrics for single sample"""
        metrics = {}
        
        ground_truth = sample['answer']
        predicted_answer = prediction['answer']
        
        # BLEU scores
        if predicted_answer and ground_truth:
            try:
                ref_tokens = [ground_truth.lower().split()]
                pred_tokens = predicted_answer.lower().split()
                
                metrics['bleu_1'] = sentence_bleu(ref_tokens, pred_tokens, weights=(1,0,0,0), smoothing_function=self.bleu_smoother)
                metrics['bleu_2'] = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5,0.5,0,0), smoothing_function=self.bleu_smoother)
                metrics['bleu_3'] = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=self.bleu_smoother)
                metrics['bleu_4'] = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=self.bleu_smoother)
            except:
                metrics.update({'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0})
        
        # ROUGE scores
        if ROUGE_AVAILABLE and predicted_answer and ground_truth:
            try:
                rouge_scores = self.rouge_scorer.score(ground_truth, predicted_answer)
                metrics['rouge_1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge_2'] = rouge_scores['rouge2'].fmeasure
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics.update({'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0})
        
        # METEOR score
        try:
            if predicted_answer and ground_truth:
                metrics['meteor'] = single_meteor_score(ground_truth, predicted_answer)
        except:
            metrics['meteor'] = 0.0
        
        # Exact match
        metrics['exact_match'] = 1.0 if predicted_answer.lower().strip() == ground_truth.lower().strip() else 0.0
        
        # Semantic similarity (simple word overlap)
        if predicted_answer and ground_truth:
            pred_words = set(predicted_answer.lower().split())
            gt_words = set(ground_truth.lower().split())
            
            if gt_words:
                metrics['word_overlap'] = len(pred_words.intersection(gt_words)) / len(gt_words)
            else:
                metrics['word_overlap'] = 0.0
        
        # Confidence and reasoning quality
        if prediction.get('confidence') is not None:
            metrics['reasoning_confidence'] = prediction['confidence']
        
        if prediction.get('reformulation_quality') is not None:
            metrics['reformulation_quality'] = prediction['reformulation_quality']
        
        return metrics
    
    def _aggregate_metrics(self, predictions: List[Dict]) -> Dict:
        """Aggregate metrics across all predictions"""
        all_metrics = {}
        
        # Collect all metric values
        metric_names = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge_1', 'rouge_2', 'rouge_l', 
                       'meteor', 'exact_match', 'word_overlap', 'reasoning_confidence', 'reformulation_quality']
        
        for metric in metric_names:
            values = [pred['metrics'].get(metric, 0.0) for pred in predictions if pred['metrics'].get(metric) is not None]
            
            if values:
                all_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                all_metrics[metric] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0, 
                    'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return all_metrics
    
    def _comparative_analysis(self, results: Dict) -> Dict:
        """Perform comparative analysis across modes"""
        analysis = {}
        
        modes = list(results.keys())
        if 'comparative_analysis' in modes:
            modes.remove('comparative_analysis')
        
        # Performance comparison
        performance_table = []
        for mode in modes:
            mode_results = results[mode]
            metrics = mode_results['metrics']
            
            row = {
                'Mode': mode,
                'BLEU-4': metrics.get('bleu_4', {}).get('mean', 0.0),
                'ROUGE-L': metrics.get('rouge_l', {}).get('mean', 0.0),
                'METEOR': metrics.get('meteor', {}).get('mean', 0.0),
                'Exact Match': metrics.get('exact_match', {}).get('mean', 0.0),
                'Word Overlap': metrics.get('word_overlap', {}).get('mean', 0.0),
                'Avg Processing Time': np.mean(mode_results['processing_times']) if mode_results['processing_times'] else 0.0,
                'Error Rate': len(mode_results['errors']) / (len(mode_results['predictions']) + len(mode_results['errors']))
            }
            
            if metrics.get('reasoning_confidence', {}).get('mean') is not None:
                row['Reasoning Confidence'] = metrics['reasoning_confidence']['mean']
            
            performance_table.append(row)
        
        analysis['performance_comparison'] = performance_table
        
        # Statistical significance testing
        significance_tests = {}
        metric_keys = ['bleu_4', 'rouge_l', 'meteor', 'exact_match', 'word_overlap']
        
        for metric in metric_keys:
            significance_tests[metric] = {}
            
            # Pairwise t-tests between modes
            for i, mode1 in enumerate(modes):
                for mode2 in modes[i+1:]:
                    values1 = [pred['metrics'].get(metric, 0.0) for pred in results[mode1]['predictions']]
                    values2 = [pred['metrics'].get(metric, 0.0) for pred in results[mode2]['predictions']]
                    
                    if values1 and values2:
                        try:
                            t_stat, p_value = stats.ttest_ind(values1, values2)
                            significance_tests[metric][f"{mode1}_vs_{mode2}"] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except:
                            significance_tests[metric][f"{mode1}_vs_{mode2}"] = {
                                't_statistic': 0.0,
                                'p_value': 1.0,
                                'significant': False
                            }
        
        analysis['significance_tests'] = significance_tests
        
        return analysis
    
    def _save_intermediate_results(self, mode: str, results: Dict):
        """Save intermediate results for each mode"""
        output_file = self.output_dir / f"{mode}_results.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {mode} results to {output_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _generate_paper_outputs(self, results: Dict):
        """Generate LaTeX tables and figures for paper"""
        self.logger.info("Generating paper outputs")
        
        # Performance comparison table
        self._generate_performance_table(results)
        
        # Performance plots
        self._generate_performance_plots(results)
        
        # Statistical significance table
        self._generate_significance_table(results)
        
        # Processing time analysis
        self._generate_timing_analysis(results)
        
        self.logger.info("Paper outputs generated successfully")
    
    def _generate_performance_table(self, results: Dict):
        """Generate LaTeX performance comparison table"""
        table_file = self.output_dir / "performance_table.tex"
        
        performance_data = results['comparative_analysis']['performance_comparison']
        
        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison of MedXplain-VQA Modes}
\label{tab:performance_comparison}
\begin{tabular}{lcccccc}
\toprule
Mode & BLEU-4 & ROUGE-L & METEOR & Exact Match & Word Overlap & Time (s) \\
\midrule
"""
        
        for row in performance_data:
            mode = row['Mode'].replace('_', '\\_')
            latex_table += f"{mode} & "
            latex_table += f"{row['BLEU-4']:.3f} & "
            latex_table += f"{row['ROUGE-L']:.3f} & "
            latex_table += f"{row['METEOR']:.3f} & "
            latex_table += f"{row['Exact Match']:.3f} & "
            latex_table += f"{row['Word Overlap']:.3f} & "
            latex_table += f"{row['Avg Processing Time']:.1f} \\\\\n"
        
        latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        self.logger.info(f"Performance table saved to {table_file}")
    
    def _generate_performance_plots(self, results: Dict):
        """Generate performance comparison plots"""
        
        # Extract data for plotting
        modes = []
        bleu_scores = []
        rouge_scores = []
        processing_times = []
        
        for mode_data in results['comparative_analysis']['performance_comparison']:
            modes.append(mode_data['Mode'])
            bleu_scores.append(mode_data['BLEU-4'])
            rouge_scores.append(mode_data['ROUGE-L'])
            processing_times.append(mode_data['Avg Processing Time'])
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # BLEU-4 comparison
        axes[0].bar(modes, bleu_scores, color='skyblue', alpha=0.8)
        axes[0].set_title('BLEU-4 Score Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('BLEU-4 Score')
        axes[0].set_ylim(0, max(bleu_scores) * 1.1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # ROUGE-L comparison
        axes[1].bar(modes, rouge_scores, color='lightcoral', alpha=0.8)
        axes[1].set_title('ROUGE-L Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('ROUGE-L Score')
        axes[1].set_ylim(0, max(rouge_scores) * 1.1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Processing time comparison
        axes[2].bar(modes, processing_times, color='lightgreen', alpha=0.8)
        axes[2].set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Time (seconds)')
        axes[2].set_ylim(0, max(processing_times) * 1.1)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "performance_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        self.logger.info("Performance plots saved")
    
    def _generate_significance_table(self, results: Dict):
        """Generate statistical significance table"""
        table_file = self.output_dir / "significance_table.tex"
        
        significance_data = results['comparative_analysis']['significance_tests']
        
        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Statistical Significance Tests (p-values)}
\label{tab:significance_tests}
\begin{tabular}{lccccc}
\toprule
Comparison & BLEU-4 & ROUGE-L & METEOR & Exact Match & Word Overlap \\
\midrule
"""
        
        # Extract comparison pairs
        if significance_data:
            first_metric = list(significance_data.keys())[0]
            comparisons = list(significance_data[first_metric].keys())
            
            for comparison in comparisons:
                comp_name = comparison.replace('_vs_', ' vs ').replace('_', '\\_')
                latex_table += f"{comp_name} & "
                
                for metric in ['bleu_4', 'rouge_l', 'meteor', 'exact_match', 'word_overlap']:
                    p_value = significance_data.get(metric, {}).get(comparison, {}).get('p_value', 1.0)
                    if p_value < 0.001:
                        latex_table += "< 0.001 & "
                    elif p_value < 0.01:
                        latex_table += "< 0.01 & "
                    elif p_value < 0.05:
                        latex_table += "< 0.05 & "
                    else:
                        latex_table += f"{p_value:.3f} & "
                
                latex_table = latex_table.rstrip(' & ') + " \\\\\n"
        
        latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        self.logger.info(f"Significance table saved to {table_file}")
    
    def _generate_timing_analysis(self, results: Dict):
        """Generate processing time analysis"""
        
        # Create timing breakdown plot
        plt.figure(figsize=(12, 8))
        
        modes = []
        times = []
        
        for mode, mode_results in results.items():
            if mode != 'comparative_analysis' and 'processing_times' in mode_results:
                modes.append(mode)
                times.append(mode_results['processing_times'])
        
        if modes and times:
            # Box plot for processing times
            plt.boxplot(times, labels=modes)
            plt.title('Processing Time Distribution by Mode', fontsize=14, fontweight='bold')
            plt.ylabel('Processing Time (seconds)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "timing_analysis.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "timing_analysis.pdf", bbox_inches='tight')
            plt.close()
        
        self.logger.info("Timing analysis saved")

def main():
    parser = argparse.ArgumentParser(description="Paper Evaluation Suite for MedXplain-VQA")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='paper_results/evaluation_suite', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--modes', nargs='+', default=['basic', 'explainable', 'enhanced', 'enhanced_bbox'], help='Evaluation modes')
    parser.add_argument('--stratified', action='store_true', help='Use stratified sampling')
    
    args = parser.parse_args()
    
    # Initialize evaluation suite
    evaluator = PaperEvaluationSuite(args.config, args.output_dir)
    
    # Load test samples
    samples = evaluator.load_test_samples(args.num_samples, args.stratified)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(samples, args.modes)
    
    # Save final results
    final_output = Path(args.output_dir) / "comprehensive_results.json"
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(evaluator._make_json_serializable(results), f, indent=2, ensure_ascii=False)
    
    print(f"✅ Paper evaluation completed. Results saved to {args.output_dir}")
    print(f"📊 Summary statistics:")
    
    for mode in args.modes:
        if mode in results:
            mode_metrics = results[mode]['metrics']
            print(f"\n{mode.upper()}:")
            print(f"  BLEU-4: {mode_metrics.get('bleu_4', {}).get('mean', 0.0):.3f}")
            print(f"  ROUGE-L: {mode_metrics.get('rouge_l', {}).get('mean', 0.0):.3f}")
            print(f"  METEOR: {mode_metrics.get('meteor', {}).get('mean', 0.0):.3f}")
            print(f"  Processing Time: {np.mean(results[mode]['processing_times']):.2f}s")

if __name__ == "__main__":
    main()
