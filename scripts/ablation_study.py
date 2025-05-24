#!/usr/bin/env python
"""
Ablation Study for MedXplain-VQA
Systematic component removal to measure contribution
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

# Evaluation metrics
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("Warning: NLTK not available. Some metrics will be skipped.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

class AblationStudy:
    """Comprehensive ablation study for MedXplain-VQA components"""
    
    def __init__(self, config_path: str, output_dir: str = "paper_results/ablation"):
        """Initialize ablation study"""
        self.config = Config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = setup_logger("ablation_study", str(self.output_dir), logging.INFO)
        
        # Define ablation configurations
        self.ablation_configs = {
            'baseline_blip': {
                'name': 'BLIP Only (Baseline)',
                'components': ['blip'],
                'description': 'Basic BLIP model without any enhancements'
            },
            'blip_gemini': {
                'name': 'BLIP + Gemini',
                'components': ['blip', 'gemini'],
                'description': 'BLIP with Gemini enhancement'
            },
            'blip_query_reform': {
                'name': 'BLIP + Query Reformulation',
                'components': ['blip', 'query_reformulation'],
                'description': 'BLIP with query reformulation'
            },
            'blip_gradcam': {
                'name': 'BLIP + Grad-CAM',
                'components': ['blip', 'gradcam'],
                'description': 'BLIP with basic Grad-CAM attention'
            },
            'blip_query_gradcam': {
                'name': 'BLIP + Query + Grad-CAM',
                'components': ['blip', 'query_reformulation', 'gradcam'],
                'description': 'BLIP with query reformulation and Grad-CAM'
            },
            'blip_query_gradcam_gemini': {
                'name': 'BLIP + Query + Grad-CAM + Gemini',
                'components': ['blip', 'query_reformulation', 'gradcam', 'gemini'],
                'description': 'All components except Chain-of-Thought'
            },
            'full_no_bbox': {
                'name': 'Full System (No BBox)',
                'components': ['blip', 'query_reformulation', 'gradcam', 'chain_of_thought', 'gemini'],
                'description': 'Full system without bounding boxes'
            },
            'full_system': {
                'name': 'Full System (With BBox)',
                'components': ['blip', 'query_reformulation', 'enhanced_gradcam', 'chain_of_thought', 'gemini'],
                'description': 'Complete MedXplain-VQA system with all enhancements'
            }
        }
        
        # Evaluation metrics
        self.bleu_smoother = SmoothingFunction().method1
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.logger.info("Ablation Study initialized")
        self.logger.info(f"Configurations: {list(self.ablation_configs.keys())}")
    
    def load_samples(self, num_samples: int = 50) -> List[Dict]:
        """Load balanced test samples for ablation study"""
        self.logger.info(f"Loading {num_samples} samples for ablation study")
        
        test_questions_file = self.config['data']['test_questions']
        
        samples = []
        with open(test_questions_file, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        
        # Balanced sampling across question types
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(all_data), min(num_samples, len(all_data)), replace=False)
        samples = [all_data[i] for i in indices]
        
        self.logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def run_ablation_study(self, samples: List[Dict]) -> Dict:
        """Run comprehensive ablation study"""
        self.logger.info(f"Starting ablation study on {len(samples)} samples")
        
        results = {}
        
        # Initialize all components once
        components = self._initialize_all_components()
        
        # Run each ablation configuration
        for config_name, config_info in self.ablation_configs.items():
            self.logger.info(f"Running ablation: {config_info['name']}")
            
            config_results = self._run_ablation_config(
                samples, config_name, config_info, components
            )
            
            results[config_name] = config_results
            
            # Save intermediate results
            self._save_config_results(config_name, config_results)
            
            self.logger.info(f"Completed: {config_info['name']}")
        
        # Comprehensive analysis
        analysis = self._analyze_ablation_results(results)
        results['analysis'] = analysis
        
        # Generate paper outputs
        self._generate_ablation_outputs(results)
        
        return results
    
    def _initialize_all_components(self) -> Dict:
        """Initialize all MedXplain components"""
        self.logger.info("Initializing all components")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        components = {
            'blip_model': BLIP2VQA(self.config, train_mode=False).to(device),
            'gemini': GeminiIntegration(self.config),
            'query_reformulator': QueryReformulator(self.config),
            'visual_context': VisualContextExtractor(self.config),
            'chain_of_thought': None,  # Will be initialized with gemini
            'enhanced_gradcam': None   # Will be initialized with blip
        }
        
        # Initialize dependent components
        components['enhanced_gradcam'] = EnhancedGradCAM(
            components['blip_model'],
            bbox_config=self.config.get('explainability', {}).get('bounding_boxes', {})
        )
        
        components['chain_of_thought'] = ChainOfThoughtGenerator(
            components['gemini'],
            self.config
        )
        
        self.logger.info("All components initialized")
        return components
    
    def _run_ablation_config(self, samples: List[Dict], config_name: str, 
                            config_info: Dict, components: Dict) -> Dict:
        """Run specific ablation configuration"""
        
        results = {
            'config_name': config_name,
            'config_info': config_info,
            'predictions': [],
            'metrics': {},
            'processing_times': [],
            'errors': []
        }
        
        for i, sample in enumerate(tqdm(samples, desc=f"Ablation: {config_name}")):
            try:
                start_time = datetime.now()
                
                # Process sample with specific component configuration
                prediction = self._process_sample_ablation(
                    sample, config_info['components'], components
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Calculate metrics
                sample_metrics = self._calculate_metrics(sample, prediction)
                
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
                self.logger.error(f"Error in {config_name}, sample {i}: {e}")
                results['errors'].append({
                    'sample_id': sample.get('image_id', f'sample_{i}'),
                    'error': str(e)
                })
        
        # Aggregate metrics
        results['metrics'] = self._aggregate_metrics(results['predictions'])
        
        return results
    
    def _process_sample_ablation(self, sample: Dict, active_components: List[str], 
                                components: Dict) -> Dict:
        """Process sample with specific component configuration"""
        from PIL import Image
        
        # Load image
        image_path = os.path.join(self.config['data']['test_images'], f"{sample['image_id']}.jpg")
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(self.config['data']['test_images'], f"{sample['image_id']}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        image = Image.open(image_path).convert('RGB')
        question = sample['question']
        
        # Start with BLIP inference (always required)
        blip_answer = components['blip_model'].predict(image, question)
        
        result = {
            'answer': blip_answer,
            'blip_answer': blip_answer,
            'processing_steps': ['blip'],
            'components_used': ['blip']
        }
        
        current_answer = blip_answer
        reformulated_question = question
        visual_context = None
        gradcam_result = None
        reasoning_result = None
        
        # Apply components based on configuration
        if 'query_reformulation' in active_components:
            reformulated = components['query_reformulator'].reformulate_question(image, question)
            reformulated_question = reformulated['reformulated_question']
            result['reformulated_question'] = reformulated_question
            result['reformulation_quality'] = reformulated['quality_score']
            result['processing_steps'].append('query_reformulation')
            result['components_used'].append('query_reformulation')
        
        if 'gradcam' in active_components:
            # Basic Grad-CAM without bounding boxes
            try:
                from src.explainability.grad_cam import GradCAM
                basic_gradcam = GradCAM(components['blip_model'])
                heatmap = basic_gradcam(image, question)
                gradcam_result = {'heatmap': heatmap, 'regions': []}
                result['attention_available'] = True
                result['processing_steps'].append('gradcam')
                result['components_used'].append('gradcam')
            except Exception as e:
                self.logger.warning(f"Basic Grad-CAM failed: {e}")
                gradcam_result = None
        
        if 'enhanced_gradcam' in active_components:
            # Enhanced Grad-CAM with bounding boxes
            try:
                gradcam_result = components['enhanced_gradcam'].analyze_image_with_question(image, question)
                result['attention_available'] = True
                result['bounding_boxes_available'] = gradcam_result.get('success', False)
                result['num_bounding_boxes'] = len(gradcam_result.get('regions', []))
                result['processing_steps'].append('enhanced_gradcam')
                result['components_used'].append('enhanced_gradcam')
            except Exception as e:
                self.logger.warning(f"Enhanced Grad-CAM failed: {e}")
                gradcam_result = None
        
        if 'chain_of_thought' in active_components:
            # Chain-of-Thought reasoning
            try:
                if visual_context is None:
                    visual_context = components['visual_context'].extract_visual_context(image, question)
                
                reasoning_result = components['chain_of_thought'].generate_reasoning_chain(
                    image, reformulated_question, blip_answer, visual_context, 
                    grad_cam_data=gradcam_result or {}
                )
                
                result['reasoning_available'] = True
                result['reasoning_confidence'] = reasoning_result.get('reasoning_chain', {}).get('overall_confidence', 0.0)
                result['reasoning_steps'] = len(reasoning_result.get('reasoning_chain', {}).get('steps', []))
                result['processing_steps'].append('chain_of_thought')
                result['components_used'].append('chain_of_thought')
                
            except Exception as e:
                self.logger.warning(f"Chain-of-Thought failed: {e}")
                reasoning_result = None
        
        if 'gemini' in active_components:
            # Gemini enhancement
            try:
                heatmap = gradcam_result.get('heatmap') if gradcam_result else None
                enhanced_answer = components['gemini'].generate_unified_answer(
                    image, reformulated_question, current_answer, heatmap=heatmap
                )
                current_answer = enhanced_answer
                result['gemini_enhanced'] = True
                result['processing_steps'].append('gemini')
                result['components_used'].append('gemini')
            except Exception as e:
                self.logger.warning(f"Gemini enhancement failed: {e}")
        
        result['answer'] = current_answer
        
        return result
    
    def _calculate_metrics(self, sample: Dict, prediction: Dict) -> Dict:
        """Calculate evaluation metrics for single sample"""
        metrics = {}
        
        ground_truth = sample['answer']
        predicted_answer = prediction['answer']
        
        # BLEU scores
        if predicted_answer and ground_truth:
            try:
                ref_tokens = [ground_truth.lower().split()]
                pred_tokens = predicted_answer.lower().split()
                
                metrics['bleu_1'] = sentence_bleu(ref_tokens, pred_tokens, weights=(1,0,0,0), smoothing_function=self.bleu_smoother)
                metrics['bleu_4'] = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=self.bleu_smoother)
            except:
                metrics.update({'bleu_1': 0.0, 'bleu_4': 0.0})
        
        # ROUGE scores
        if ROUGE_AVAILABLE and predicted_answer and ground_truth:
            try:
                rouge_scores = self.rouge_scorer.score(ground_truth, predicted_answer)
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics['rouge_l'] = 0.0
        
        # METEOR score
        try:
            if predicted_answer and ground_truth:
                metrics['meteor'] = single_meteor_score(ground_truth, predicted_answer)
        except:
            metrics['meteor'] = 0.0
        
        # Exact match
        metrics['exact_match'] = 1.0 if predicted_answer.lower().strip() == ground_truth.lower().strip() else 0.0
        
        # Word overlap
        if predicted_answer and ground_truth:
            pred_words = set(predicted_answer.lower().split())
            gt_words = set(ground_truth.lower().split())
            
            if gt_words:
                metrics['word_overlap'] = len(pred_words.intersection(gt_words)) / len(gt_words)
            else:
                metrics['word_overlap'] = 0.0
        
        # Component-specific metrics
        if prediction.get('reasoning_confidence'):
            metrics['reasoning_confidence'] = prediction['reasoning_confidence']
        
        if prediction.get('reformulation_quality'):
            metrics['reformulation_quality'] = prediction['reformulation_quality']
        
        return metrics
    
    def _aggregate_metrics(self, predictions: List[Dict]) -> Dict:
        """Aggregate metrics across all predictions"""
        all_metrics = {}
        
        metric_names = ['bleu_1', 'bleu_4', 'rouge_l', 'meteor', 'exact_match', 'word_overlap', 
                       'reasoning_confidence', 'reformulation_quality']
        
        for metric in metric_names:
            values = [pred['metrics'].get(metric, 0.0) for pred in predictions if pred['metrics'].get(metric) is not None]
            
            if values:
                all_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'count': len(values)
                }
            else:
                all_metrics[metric] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0, 'count': 0
                }
        
        return all_metrics
    
    def _analyze_ablation_results(self, results: Dict) -> Dict:
        """Analyze ablation study results"""
        analysis = {}
        
        # Performance progression
        config_order = ['baseline_blip', 'blip_gemini', 'blip_query_reform', 'blip_gradcam', 
                       'blip_query_gradcam', 'blip_query_gradcam_gemini', 'full_no_bbox', 'full_system']
        
        progression_data = []
        baseline_scores = None
        
        for config_name in config_order:
            if config_name in results:
                config_data = results[config_name]
                metrics = config_data['metrics']
                
                row = {
                    'Configuration': self.ablation_configs[config_name]['name'],
                    'Components': ' + '.join(self.ablation_configs[config_name]['components']),
                    'BLEU-4': metrics.get('bleu_4', {}).get('mean', 0.0),
                    'ROUGE-L': metrics.get('rouge_l', {}).get('mean', 0.0),
                    'METEOR': metrics.get('meteor', {}).get('mean', 0.0),
                    'Exact Match': metrics.get('exact_match', {}).get('mean', 0.0),
                    'Processing Time': np.mean(config_data['processing_times']) if config_data['processing_times'] else 0.0,
                    'Error Rate': len(config_data['errors']) / (len(config_data['predictions']) + len(config_data['errors'])) if config_data['predictions'] or config_data['errors'] else 0.0
                }
                
                # Calculate improvement over baseline
                if config_name == 'baseline_blip':
                    baseline_scores = {
                        'BLEU-4': row['BLEU-4'],
                        'ROUGE-L': row['ROUGE-L'],
                        'METEOR': row['METEOR'],
                        'Exact Match': row['Exact Match']
                    }
                    row.update({
                        'BLEU-4 Δ': 0.0,
                        'ROUGE-L Δ': 0.0,
                        'METEOR Δ': 0.0,
                        'Exact Match Δ': 0.0
                    })
                else:
                    if baseline_scores:
                        row.update({
                            'BLEU-4 Δ': row['BLEU-4'] - baseline_scores['BLEU-4'],
                            'ROUGE-L Δ': row['ROUGE-L'] - baseline_scores['ROUGE-L'],
                            'METEOR Δ': row['METEOR'] - baseline_scores['METEOR'],
                            'Exact Match Δ': row['Exact Match'] - baseline_scores['Exact Match']
                        })
                
                progression_data.append(row)
        
        analysis['progression'] = progression_data
        
        # Component contribution analysis
        component_contributions = self._analyze_component_contributions(results, config_order)
        analysis['component_contributions'] = component_contributions
        
        # Statistical significance
        significance_tests = self._perform_significance_tests(results, config_order)
        analysis['significance_tests'] = significance_tests
        
        return analysis
    
    def _analyze_component_contributions(self, results: Dict, config_order: List[str]) -> Dict:
        """Analyze individual component contributions"""
        contributions = {}
        
        # Define component addition steps
        component_steps = [
            ('gemini', 'baseline_blip', 'blip_gemini'),
            ('query_reformulation', 'baseline_blip', 'blip_query_reform'),
            ('gradcam', 'baseline_blip', 'blip_gradcam'),
            ('gemini', 'blip_query_gradcam', 'blip_query_gradcam_gemini'),
            ('chain_of_thought', 'blip_query_gradcam_gemini', 'full_no_bbox'),
            ('enhanced_gradcam', 'full_no_bbox', 'full_system')
        ]
        
        for component, before_config, after_config in component_steps:
            if before_config in results and after_config in results:
                before_metrics = results[before_config]['metrics']
                after_metrics = results[after_config]['metrics']
                
                contribution = {
                    'component': component,
                    'bleu_4_improvement': after_metrics.get('bleu_4', {}).get('mean', 0.0) - before_metrics.get('bleu_4', {}).get('mean', 0.0),
                    'rouge_l_improvement': after_metrics.get('rouge_l', {}).get('mean', 0.0) - before_metrics.get('rouge_l', {}).get('mean', 0.0),
                    'meteor_improvement': after_metrics.get('meteor', {}).get('mean', 0.0) - before_metrics.get('meteor', {}).get('mean', 0.0),
                    'exact_match_improvement': after_metrics.get('exact_match', {}).get('mean', 0.0) - before_metrics.get('exact_match', {}).get('mean', 0.0)
                }
                
                contributions[component] = contribution
        
        return contributions
    
    def _perform_significance_tests(self, results: Dict, config_order: List[str]) -> Dict:
        """Perform statistical significance tests"""
        significance_tests = {}
        
        # Compare sequential configurations
        for i in range(len(config_order) - 1):
            config1 = config_order[i]
            config2 = config_order[i + 1]
            
            if config1 in results and config2 in results:
                for metric in ['bleu_4', 'rouge_l', 'meteor', 'exact_match']:
                    values1 = [pred['metrics'].get(metric, 0.0) for pred in results[config1]['predictions']]
                    values2 = [pred['metrics'].get(metric, 0.0) for pred in results[config2]['predictions']]
                    
                    if values1 and values2:
                        try:
                            t_stat, p_value = stats.ttest_rel(values1, values2)  # Paired t-test
                            
                            test_key = f"{config1}_vs_{config2}"
                            if test_key not in significance_tests:
                                significance_tests[test_key] = {}
                            
                            significance_tests[test_key][metric] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'highly_significant': p_value < 0.01
                            }
                        except:
                            pass
        
        return significance_tests
    
    def _save_config_results(self, config_name: str, results: Dict):
        """Save intermediate results for each configuration"""
        output_file = self.output_dir / f"ablation_{config_name}.json"
        
        # Convert for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {config_name} results to {output_file}")
    
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
    
    def _generate_ablation_outputs(self, results: Dict):
        """Generate paper outputs for ablation study"""
        self.logger.info("Generating ablation study outputs")
        
        # Ablation table
        self._generate_ablation_table(results)
        
        # Component contribution plot
        self._generate_contribution_plot(results)
        
        # Performance progression plot
        self._generate_progression_plot(results)
        
        # Statistical significance table
        self._generate_ablation_significance_table(results)
        
        self.logger.info("Ablation outputs generated")
    
    def _generate_ablation_table(self, results: Dict):
        """Generate LaTeX ablation study table"""
        table_file = self.output_dir / "ablation_table.tex"
        
        progression_data = results['analysis']['progression']
        
        latex_table = r"""
\begin{table*}[htbp]
\centering
\caption{Ablation Study Results: Component-wise Performance Analysis}
\label{tab:ablation_study}
\begin{tabular}{lcccccc}
\toprule
Configuration & BLEU-4 & ROUGE-L & METEOR & Exact Match & Time (s) & Error Rate \\
\midrule
"""
        
        for row in progression_data:
            config_name = row['Configuration'].replace('_', '\\_')
            latex_table += f"{config_name} & "
            latex_table += f"{row['BLEU-4']:.3f} & "
            latex_table += f"{row['ROUGE-L']:.3f} & "
            latex_table += f"{row['METEOR']:.3f} & "
            latex_table += f"{row['Exact Match']:.3f} & "
            latex_table += f"{row['Processing Time']:.1f} & "
            latex_table += f"{row['Error Rate']:.3f} \\\\\n"
        
        latex_table += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
        
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        self.logger.info(f"Ablation table saved to {table_file}")
    
    def _generate_contribution_plot(self, results: Dict):
        """Generate component contribution plot"""
        
        contributions = results['analysis']['component_contributions']
        
        if not contributions:
            self.logger.warning("No contribution data available for plotting")
            return
        
        components = list(contributions.keys())
        bleu_improvements = [contributions[comp]['bleu_4_improvement'] for comp in components]
        rouge_improvements = [contributions[comp]['rouge_l_improvement'] for comp in components]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # BLEU-4 improvements
        bars1 = ax1.bar(components, bleu_improvements, color='skyblue', alpha=0.8)
        ax1.set_title('BLEU-4 Improvement by Component', fontsize=14, fontweight='bold')
        ax1.set_ylabel('BLEU-4 Improvement')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, bleu_improvements):
            if value >= 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ROUGE-L improvements
        bars2 = ax2.bar(components, rouge_improvements, color='lightcoral', alpha=0.8)
        ax2.set_title('ROUGE-L Improvement by Component', fontsize=14, fontweight='bold')
        ax2.set_ylabel('ROUGE-L Improvement')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, rouge_improvements):
            if value >= 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "component_contributions.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "component_contributions.pdf", bbox_inches='tight')
        plt.close()
        
        self.logger.info("Component contribution plots saved")
    
    def _generate_progression_plot(self, results: Dict):
        """Generate performance progression plot"""
        
        progression_data = results['analysis']['progression']
        
        configurations = [row['Configuration'] for row in progression_data]
        bleu_scores = [row['BLEU-4'] for row in progression_data]
        rouge_scores = [row['ROUGE-L'] for row in progression_data]
        
        plt.figure(figsize=(14, 8))
        
        x_pos = np.arange(len(configurations))
        
        plt.plot(x_pos, bleu_scores, 'o-', linewidth=2.5, markersize=8, label='BLEU-4', color='blue')
        plt.plot(x_pos, rouge_scores, 's-', linewidth=2.5, markersize=8, label='ROUGE-L', color='red')
        
        plt.xlabel('Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Performance Progression in Ablation Study', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, [config.replace(' ', '\n') for config in configurations], rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (bleu, rouge) in enumerate(zip(bleu_scores, rouge_scores)):
            plt.annotate(f'{bleu:.3f}', (i, bleu), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(f'{rouge:.3f}', (i, rouge), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_progression.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "performance_progression.pdf", bbox_inches='tight')
        plt.close()
        
        self.logger.info("Performance progression plot saved")
    
    def _generate_ablation_significance_table(self, results: Dict):
        """Generate statistical significance table for ablation study"""
        table_file = self.output_dir / "ablation_significance.tex"
        
        significance_data = results['analysis']['significance_tests']
        
        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Statistical Significance Tests for Ablation Study}
\label{tab:ablation_significance}
\begin{tabular}{lccccc}
\toprule
Comparison & BLEU-4 & ROUGE-L & METEOR & Exact Match \\
\midrule
"""
        
        for comparison, test_results in significance_data.items():
            comp_name = comparison.replace('_vs_', ' $\\rightarrow$ ').replace('_', '\\_')
            latex_table += f"{comp_name} & "
            
            for metric in ['bleu_4', 'rouge_l', 'meteor', 'exact_match']:
                if metric in test_results:
                    p_value = test_results[metric]['p_value']
                    if p_value < 0.001:
                        latex_table += "***  & "
                    elif p_value < 0.01:
                        latex_table += "** & "
                    elif p_value < 0.05:
                        latex_table += "* & "
                    else:
                        latex_table += "ns & "
                else:
                    latex_table += "- & "
            
            latex_table = latex_table.rstrip(' & ') + " \\\\\n"
        
        latex_table += r"""
\bottomrule
\multicolumn{5}{l}{\footnotesize *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant}
\end{tabular}
\end{table}
"""
        
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        self.logger.info(f"Ablation significance table saved to {table_file}")

def main():
    parser = argparse.ArgumentParser(description="Ablation Study for MedXplain-VQA")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='paper_results/ablation', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of test samples')
    
    args = parser.parse_args()
    
    # Initialize ablation study
    ablation_study = AblationStudy(args.config, args.output_dir)
    
    # Load test samples
    samples = ablation_study.load_samples(args.num_samples)
    
    # Run ablation study
    results = ablation_study.run_ablation_study(samples)
    
    # Save final results
    final_output = Path(args.output_dir) / "ablation_results.json"
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(ablation_study._make_json_serializable(results), f, indent=2, ensure_ascii=False)
    
    print(f"✅ Ablation study completed. Results saved to {args.output_dir}")
    print(f"📊 Component contributions:")
    
    if 'analysis' in results and 'component_contributions' in results['analysis']:
        for component, contribution in results['analysis']['component_contributions'].items():
            print(f"  {component}: BLEU-4 +{contribution['bleu_4_improvement']:.3f}, ROUGE-L +{contribution['rouge_l_improvement']:.3f}")

if __name__ == "__main__":
    main()
