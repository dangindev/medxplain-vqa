#!/usr/bin/env python
"""
🎯 FINAL PAPER EVALUATION: 50 samples + torch compatibility + paper-ready results
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# Fix torch compatibility issue
import torch
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

class PaperEvaluationFinal:
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = base_data_dir
        
        # Load semantic model with compatibility fix
        self.semantic_model = self._load_semantic_model_safe()
        
        # Define mode configurations
        self.mode_configs = {
            'basic': {
                'dir': 'eval_basic',
                'name': 'BLIP + Gemini',
                'short_name': 'Basic',
                'description': 'Basic VQA with LLM enhancement'
            },
            'explainable': {
                'dir': 'eval_explainable', 
                'name': 'BLIP + Query Reform + GradCAM',
                'short_name': 'Explainable',
                'description': 'Explainable VQA with query reformulation'
            },
            'explainable_bbox': {
                'dir': 'eval_bbox',
                'name': 'BLIP + ... + Bounding Boxes',
                'short_name': 'ExplainableBBox',
                'description': 'Explainable VQA with bounding box attention'
            },
            'enhanced': {
                'dir': 'eval_enhanced',
                'name': 'BLIP + ... + Chain-of-Thought',
                'short_name': 'Enhanced',
                'description': 'Enhanced VQA with reasoning chains'
            },
            'enhanced_bbox': {
                'dir': 'eval_full',
                'name': 'MedXplain-VQA (Full)',
                'short_name': 'Full',
                'description': 'Complete system with all components'
            }
        }
    
    def _load_semantic_model_safe(self):
        """Load semantic model with full compatibility"""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic similarity model loaded successfully")
            return model
            
        except Exception as e:
            print(f"⚠️ Could not load sentence-transformers: {e}")
            print("📝 Will use fallback similarity methods")
            return None
    
    def load_all_results(self):
        """Load results từ tất cả modes"""
        all_results = {}
        
        for mode_key, config in self.mode_configs.items():
            results_dir = os.path.join(self.base_data_dir, config['dir'])
            
            if not os.path.exists(results_dir):
                print(f"⚠️ Directory not found: {results_dir}")
                continue
                
            mode_results = []
            json_files = list(Path(results_dir).glob("*.json"))
            
            print(f"📂 Loading {len(json_files)} results from {config['name']}...")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        result['mode_key'] = mode_key
                        mode_results.append(result)
                except Exception as e:
                    print(f"❌ Error loading {json_file}: {e}")
                    continue
            
            all_results[mode_key] = mode_results
            print(f"✅ Loaded {len(mode_results)} results for {config['name']}")
        
        return all_results
    
    def compute_answer_relevance_score(self, predicted, ground_truth):
        """Medical domain appropriate answer relevance"""
        if not predicted.strip() or not ground_truth.strip():
            return 0.0
        
        pred_lower = predicted.lower()
        gt_lower = ground_truth.lower()
        
        # Exact match bonus
        if pred_lower == gt_lower:
            return 1.0
        
        # Substring containment (either direction)
        if gt_lower in pred_lower or pred_lower in gt_lower:
            return 0.8
        
        # Medical keywords overlap
        medical_keywords = [
            'cell', 'tissue', 'lesion', 'structure', 'gland', 'follicle',
            'tumor', 'carcinoma', 'melanoma', 'nevus', 'inflammation',
            'dermatitis', 'fibrosis', 'hyperplasia', 'dysplasia',
            'benign', 'malignant', 'pathology', 'diagnosis', 'demodex',
            'folliculorum', 'sebaceous', 'keratin', 'epithelial', 'stromal'
        ]
        
        pred_keywords = [kw for kw in medical_keywords if kw in pred_lower]
        gt_keywords = [kw for kw in medical_keywords if kw in gt_lower]
        
        if pred_keywords and gt_keywords:
            keyword_overlap = len(set(pred_keywords).intersection(set(gt_keywords)))
            keyword_union = len(set(pred_keywords).union(set(gt_keywords)))
            if keyword_union > 0:
                return 0.3 + 0.5 * (keyword_overlap / keyword_union)
        
        # Jaccard similarity for words
        pred_words = set(pred_lower.split())
        gt_words = set(gt_lower.split())
        
        intersection = len(pred_words.intersection(gt_words))
        union = len(pred_words.union(gt_words))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        return min(0.7, jaccard_score * 0.8)
    
    def compute_semantic_similarity(self, predicted, ground_truth):
        """Semantic similarity với fallback support"""
        if self.semantic_model:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                
                pred_emb = self.semantic_model.encode([predicted.lower()])
                gt_emb = self.semantic_model.encode([ground_truth.lower()])
                
                similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
                return float(similarity)
                
            except Exception as e:
                print(f"Semantic similarity error: {e}")
                
        # Fallback to improved lexical similarity
        return self.compute_answer_relevance_score(predicted, ground_truth)
    
    def extract_medical_concepts(self, text):
        """Enhanced medical concept extraction"""
        text_lower = text.lower()
        
        medical_patterns = {
            'pathology_terms': [
                r'\b(?:carcinoma|melanoma|sarcoma|lymphoma|leukemia)\b',
                r'\b(?:adenoma|papilloma|fibroma|lipoma)\b',
                r'\b(?:nevus|mole|lesion|tumor|mass)\b',
                r'\b(?:demodex|folliculorum|sebaceous|keratin)\b'
            ],
            'anatomical_terms': [
                r'\b(?:epidermis|dermis|subcutaneous|follicle)\b', 
                r'\b(?:gland|duct|vessel|nerve|muscle)\b',
                r'\b(?:thyroid|parathyroid|endocrine|exocrine)\b'
            ],
            'condition_terms': [
                r'\b(?:inflammation|hyperplasia|dysplasia|metaplasia)\b',
                r'\b(?:fibrosis|sclerosis|atrophy|necrosis)\b',
                r'\b(?:benign|malignant|invasive|metastatic)\b'
            ],
            'cellular_terms': [
                r'\b(?:epithelial|stromal|lymphoid|neural)\b',
                r'\b(?:cuboidal|columnar|squamous|basal)\b',
                r'\b(?:cell|cells|tissue|tissues)\b'
            ]
        }
        
        concepts = []
        for category, patterns in medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                concepts.extend([(match, category) for match in matches])
        
        return list(set(concepts))
    
    def compute_clinical_accuracy_score(self, predicted, ground_truth):
        """Clinical accuracy assessment"""
        pred_concepts = self.extract_medical_concepts(predicted)
        gt_concepts = self.extract_medical_concepts(ground_truth)
        
        if not gt_concepts:
            return 1.0 if not pred_concepts else 0.5
        
        pred_terms = set([concept[0] for concept in pred_concepts])
        gt_terms = set([concept[0] for concept in gt_concepts])
        
        # Exact medical term matches
        exact_matches = len(pred_terms.intersection(gt_terms))
        
        if exact_matches > 0:
            return min(1.0, exact_matches / len(gt_terms) + 0.3)
        
        # Category-level matches
        pred_categories = set([concept[1] for concept in pred_concepts])
        gt_categories = set([concept[1] for concept in gt_concepts])
        
        category_matches = len(pred_categories.intersection(gt_categories))
        
        if category_matches > 0:
            return min(0.7, category_matches / len(gt_categories) * 0.6)
        
        return 0.1  # Some credit for medical content
    
    def analyze_mode_performance(self, mode_results):
        """Comprehensive performance analysis"""
        if not mode_results:
            return {}
        
        total_samples = len(mode_results)
        successful_samples = sum(1 for r in mode_results if r.get('success', False))
        success_rate = successful_samples / total_samples if total_samples > 0 else 0
        
        # Multi-dimensional evaluation
        answer_relevance_scores = []
        semantic_similarity_scores = []
        clinical_accuracy_scores = []
        
        for result in mode_results:
            if result.get('success', False):
                predicted = result.get('unified_answer', '')
                ground_truth = result.get('ground_truth', '')
                
                if predicted and ground_truth:
                    relevance_score = self.compute_answer_relevance_score(predicted, ground_truth)
                    answer_relevance_scores.append(relevance_score)
                    
                    semantic_score = self.compute_semantic_similarity(predicted, ground_truth)
                    semantic_similarity_scores.append(semantic_score)
                    
                    clinical_score = self.compute_clinical_accuracy_score(predicted, ground_truth)
                    clinical_accuracy_scores.append(clinical_score)
        
        # Component analysis
        reformulation_qualities = []
        for result in mode_results:
            if 'reformulation_quality' in result:
                reformulation_qualities.append(result['reformulation_quality'])
        
        attention_metrics = self.analyze_attention_quality(mode_results)
        reasoning_metrics = self.analyze_reasoning_quality(mode_results)
        
        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': success_rate,
            'answer_relevance': {
                'mean': np.mean(answer_relevance_scores) if answer_relevance_scores else 0,
                'std': np.std(answer_relevance_scores) if answer_relevance_scores else 0,
                'scores': answer_relevance_scores
            },
            'semantic_similarity': {
                'mean': np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0,
                'std': np.std(semantic_similarity_scores) if semantic_similarity_scores else 0,
                'scores': semantic_similarity_scores
            },
            'clinical_accuracy': {
                'mean': np.mean(clinical_accuracy_scores) if clinical_accuracy_scores else 0,
                'std': np.std(clinical_accuracy_scores) if clinical_accuracy_scores else 0,
                'scores': clinical_accuracy_scores
            },
            'reformulation_quality': {
                'mean': np.mean(reformulation_qualities) if reformulation_qualities else 0,
                'std': np.std(reformulation_qualities) if reformulation_qualities else 0,
                'count': len(reformulation_qualities)
            },
            'attention_metrics': attention_metrics,
            'reasoning_metrics': reasoning_metrics
        }
    
    def analyze_attention_quality(self, mode_results):
        """Analyze attention/bounding box quality"""
        bbox_counts = []
        avg_attention_scores = []
        max_attention_scores = []
        
        for result in mode_results:
            if result.get('bbox_regions_count', 0) > 0:
                bbox_counts.append(result['bbox_regions_count'])
                
                if 'bounding_box_analysis' in result:
                    bbox_analysis = result['bounding_box_analysis']
                    avg_attention_scores.append(bbox_analysis.get('average_attention_score', 0))
                    max_attention_scores.append(bbox_analysis.get('max_attention_score', 0))
        
        return {
            'bbox_detection_rate': len(bbox_counts) / len(mode_results) if mode_results else 0,
            'avg_regions_per_image': np.mean(bbox_counts) if bbox_counts else 0,
            'avg_attention_score': np.mean(avg_attention_scores) if avg_attention_scores else 0,
            'max_attention_score': np.mean(max_attention_scores) if max_attention_scores else 0,
            'total_images_with_bbox': len(bbox_counts)
        }
    
    def analyze_reasoning_quality(self, mode_results):
        """Analyze chain-of-thought reasoning quality"""
        reasoning_confidences = []
        reasoning_step_counts = []
        reasoning_flows = []
        
        for result in mode_results:
            if 'reasoning_analysis' in result:
                reasoning = result['reasoning_analysis']
                reasoning_confidences.append(reasoning.get('reasoning_confidence', 0))
                reasoning_step_counts.append(reasoning.get('reasoning_steps_count', 0))
                reasoning_flows.append(reasoning.get('reasoning_flow', 'unknown'))
        
        flow_counts = {}
        for flow in reasoning_flows:
            flow_counts[flow] = flow_counts.get(flow, 0) + 1
        
        return {
            'reasoning_usage_rate': len(reasoning_confidences) / len(mode_results) if mode_results else 0,
            'avg_reasoning_confidence': np.mean(reasoning_confidences) if reasoning_confidences else 0,
            'avg_reasoning_steps': np.mean(reasoning_step_counts) if reasoning_step_counts else 0,
            'reasoning_flow_distribution': flow_counts,
            'total_with_reasoning': len(reasoning_confidences)
        }
    
    def create_paper_ablation_table(self, all_analysis):
        """Create paper-ready ablation study table"""
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        table_data = []
        
        for mode_key in ablation_order:
            if mode_key not in all_analysis:
                continue
                
            analysis = all_analysis[mode_key]
            config = self.mode_configs[mode_key]
            
            row = {
                'Method': config['short_name'],
                'Success Rate (%)': f"{analysis['success_rate']*100:.1f}",
                'Answer Relevance': f"{analysis['answer_relevance']['mean']:.3f}",
                'Semantic Similarity': f"{analysis['semantic_similarity']['mean']:.3f}",
                'Clinical Accuracy': f"{analysis['clinical_accuracy']['mean']:.3f}",
                'Query Quality': f"{analysis['reformulation_quality']['mean']:.3f}" if analysis['reformulation_quality']['count'] > 0 else "—",
                'Attention Score': f"{analysis['attention_metrics']['avg_attention_score']:.3f}" if analysis['attention_metrics']['total_images_with_bbox'] > 0 else "—",
                'Reasoning Conf.': f"{analysis['reasoning_metrics']['avg_reasoning_confidence']:.3f}" if analysis['reasoning_metrics']['total_with_reasoning'] > 0 else "—"
            }
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def run_blip_baseline(self, num_samples=None):
        """Run BLIP-only baseline comparison"""
        print("🔬 Running BLIP-only baseline...")
        
        # Load config and model
        config = Config("configs/config.yaml")
        logger = setup_logger('baseline_comparison', config['logging']['save_dir'])
        
        # Load BLIP model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            model = BLIP2VQA(config, train_mode=False)
            model.device = device
            
            model_path = "checkpoints/blip/checkpoints/best_hf_model"
            if os.path.isdir(model_path):
                model.model = type(model.model).from_pretrained(model_path)
                model.model.to(device)
                logger.info("Loaded BLIP model for baseline")
            
            model.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading BLIP model for baseline: {e}")
            return None
        
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
        
        if num_samples:
            selected_questions = questions[:num_samples]
        else:
            # Match number of samples from eval data
            selected_questions = questions[:5]  # Default to 5 for consistency
        
        # Process samples
        baseline_results = []
        
        for item in selected_questions:
            image_id = item['image_id']
            
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = Path(test_images_dir) / f"{image_id}{ext}"
                if img_path.exists():
                    try:
                        from PIL import Image
                        image = Image.open(img_path).convert('RGB')
                        
                        blip_answer = model.predict(image, item['question'])
                        
                        # Compute similarity
                        similarity_score = self.compute_semantic_similarity(blip_answer, item['answer'])
                        
                        baseline_results.append({
                            'sample_id': image_id,
                            'question': item['question'],
                            'ground_truth': item['answer'],
                            'blip_only_answer': blip_answer,
                            'similarity_score': similarity_score
                        })
                        
                        print(f"✅ Baseline {image_id}: {similarity_score:.3f}")
                        
                    except Exception as e:
                        print(f"❌ Error processing baseline {image_id}: {e}")
                        continue
                    break
        
        # Compute baseline summary
        similarity_scores = [r['similarity_score'] for r in baseline_results]
        
        baseline_summary = {
            'total_samples': len(baseline_results),
            'avg_similarity': np.mean(similarity_scores) if similarity_scores else 0,
            'std_similarity': np.std(similarity_scores) if similarity_scores else 0
        }
        
        print(f"📊 BLIP-only baseline: {baseline_summary['avg_similarity']:.3f} ± {baseline_summary['std_similarity']:.3f}")
        
        return baseline_summary
    
    def generate_paper_results(self, output_dir="data/paper_results_final"):
        """Generate complete paper-ready results"""
        print("🚀 Generating Final Paper Results")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all evaluation results
        all_results = self.load_all_results()
        
        if not all_results:
            print("❌ No evaluation results found")
            return
        
        # Analyze each mode
        print("\n📊 Analyzing all methods...")
        all_analysis = {}
        
        for mode_key, mode_results in all_results.items():
            config = self.mode_configs[mode_key]
            print(f"\n🔍 {config['name']} ({len(mode_results)} samples)...")
            
            analysis = self.analyze_mode_performance(mode_results)
            all_analysis[mode_key] = analysis
            
            print(f"  Success Rate: {analysis['success_rate']*100:.1f}%")
            print(f"  Answer Relevance: {analysis['answer_relevance']['mean']:.3f}")
            print(f"  Semantic Similarity: {analysis['semantic_similarity']['mean']:.3f}")
            print(f"  Clinical Accuracy: {analysis['clinical_accuracy']['mean']:.3f}")
        
        # Run BLIP baseline
        print(f"\n🔬 Running BLIP-only baseline comparison...")
        baseline_summary = self.run_blip_baseline(num_samples=len(list(all_results.values())[0]))
        
        # Create paper tables
        print(f"\n📋 Creating paper tables...")
        ablation_df = self.create_paper_ablation_table(all_analysis)
        
        # Add baseline row
        if baseline_summary:
            baseline_row = {
                'Method': 'BLIP-only',
                'Success Rate (%)': '100.0',
                'Answer Relevance': '—',
                'Semantic Similarity': f"{baseline_summary['avg_similarity']:.3f}",
                'Clinical Accuracy': '—',
                'Query Quality': '—',
                'Attention Score': '—',
                'Reasoning Conf.': '—'
            }
            ablation_df = pd.concat([
                pd.DataFrame([baseline_row]), 
                ablation_df
            ], ignore_index=True)
        
        # Generate LaTeX table
        latex_table = ablation_df.to_latex(
            index=False,
            escape=False,
            caption="MedXplain-VQA Ablation Study: Component Performance Analysis",
            label="tab:medxplain_ablation_study",
            column_format="l|c|c|c|c|c|c|c"
        )
        
        # Save LaTeX table
        latex_file = os.path.join(output_dir, "paper_ablation_table.tex")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        # Create performance plots
        self.create_paper_plots(all_analysis, baseline_summary, output_dir)
        
        # Save complete results
        results_file = os.path.join(output_dir, "paper_evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_results': all_analysis,
                'baseline_results': baseline_summary,
                'summary_table': ablation_df.to_dict('records')
            }, f, indent=2, default=self._json_serialize)
        
        print(f"\n🎉 Paper results generated successfully!")
        print(f"📊 Files created:")
        print(f"  • paper_ablation_table.tex - LaTeX table for paper")
        print(f"  • paper_performance_plots.png - Performance visualization")
        print(f"  • paper_evaluation_results.json - Complete data")
        print(f"📁 Output directory: {output_dir}")
        
        # Print paper table
        print(f"\n📋 PAPER ABLATION TABLE:")
        print(ablation_df.to_string(index=False))
        
        # Find best method
        best_clinical = max(all_analysis.values(), key=lambda x: x['clinical_accuracy']['mean'])
        best_mode = [k for k, v in all_analysis.items() if v == best_clinical][0]
        best_config = self.mode_configs[best_mode]
        
        print(f"\n🏆 BEST PERFORMING METHOD:")
        print(f"Method: {best_config['name']}")
        print(f"Clinical Accuracy: {best_clinical['clinical_accuracy']['mean']:.3f}")
        print(f"Semantic Similarity: {best_clinical['semantic_similarity']['mean']:.3f}")
        print(f"Success Rate: {best_clinical['success_rate']*100:.1f}%")
        
        return all_analysis, ablation_df, baseline_summary
    
    def create_paper_plots(self, all_analysis, baseline_summary, output_dir):
        """Create publication-quality plots"""
        
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        methods = ['BLIP-only']  # Start with baseline
        semantic_scores = [baseline_summary['avg_similarity'] if baseline_summary else 0]
        clinical_scores = [0]  # Baseline doesn't have clinical accuracy
        relevance_scores = [0]  # Baseline doesn't have relevance score
        
        for mode_key in ablation_order:
            if mode_key in all_analysis:
                analysis = all_analysis[mode_key]
                config = self.mode_configs[mode_key]
                
                methods.append(config['short_name'])
                semantic_scores.append(analysis['semantic_similarity']['mean'])
                clinical_scores.append(analysis['clinical_accuracy']['mean'])
                relevance_scores.append(analysis['answer_relevance']['mean'])
        
        # Create publication-quality plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x_pos = np.arange(len(methods))
        
        # Plot 1: Semantic Similarity
        bars1 = ax1.bar(x_pos, semantic_scores, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        ax1.set_xlabel('Method', fontsize=12)
        ax1.set_ylabel('Semantic Similarity Score', fontsize=12)
        ax1.set_title('Semantic Similarity Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(semantic_scores) * 1.1)
        
        for bar, score in zip(bars1, semantic_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Clinical Accuracy
        bars2 = ax2.bar(x_pos[1:], clinical_scores[1:], color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Clinical Accuracy Score', fontsize=12)
        ax2.set_title('Clinical Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos[1:])
        ax2.set_xticklabels(methods[1:], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(clinical_scores[1:]) * 1.1 if clinical_scores[1:] else 1)
        
        for bar, score in zip(bars2, clinical_scores[1:]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Answer Relevance
        bars3 = ax3.bar(x_pos[1:], relevance_scores[1:], color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
        ax3.set_xlabel('Method', fontsize=12)
        ax3.set_ylabel('Answer Relevance Score', fontsize=12)
        ax3.set_title('Answer Relevance Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos[1:])
        ax3.set_xticklabels(methods[1:], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, max(relevance_scores[1:]) * 1.1 if relevance_scores[1:] else 1)
        
        for bar, score in zip(bars3, relevance_scores[1:]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Combined Performance (exclude baseline for clinical+relevance)
        combined_scores = [(s + c + r) / 3 for s, c, r in zip(semantic_scores[1:], clinical_scores[1:], relevance_scores[1:])]
        bars4 = ax4.bar(x_pos[1:], combined_scores, color='gold', alpha=0.8, edgecolor='orange', linewidth=1)
        ax4.set_xlabel('Method', fontsize=12)
        ax4.set_ylabel('Combined Performance Score', fontsize=12)
        ax4.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos[1:])
        ax4.set_xticklabels(methods[1:], rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, max(combined_scores) * 1.1 if combined_scores else 1)
        
        for bar, score in zip(bars4, combined_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "paper_performance_plots.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Publication-quality plots saved")
    
    def _json_serialize(self, obj):
        """JSON serialization helper"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

def main():
    print("🎯 MedXplain-VQA Final Paper Evaluation")
    print("="*45)
    
    # Initialize evaluator
    evaluator = PaperEvaluationFinal()
    
    # Generate paper results
    try:
        all_analysis, ablation_df, baseline_summary = evaluator.generate_paper_results()
        print("\n✅ Final paper evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during final evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
