#!/usr/bin/env python
"""
🎯 MedXplain-VQA Evaluation Results Analyzer - Final Version
Analyze results từ 50 samples và generate paper-ready tables
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

class EvaluationAnalyzer:
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = base_data_dir
        self.semantic_model = self._load_semantic_model_safe()
        
        # Mode configurations
        self.mode_configs = {
            'basic': {
                'dir': 'eval_basic',
                'name': 'BLIP + Gemini',
                'description': 'Basic VQA with LLM enhancement'
            },
            'explainable': {
                'dir': 'eval_explainable', 
                'name': 'BLIP + Query Reform + GradCAM',
                'description': 'Explainable VQA with query reformulation'
            },
            'explainable_bbox': {
                'dir': 'eval_bbox',
                'name': 'BLIP + ... + Bounding Boxes',
                'description': 'Explainable VQA with bounding box attention'
            },
            'enhanced': {
                'dir': 'eval_enhanced',
                'name': 'BLIP + ... + Chain-of-Thought',
                'description': 'Enhanced VQA with reasoning chains'
            },
            'enhanced_bbox': {
                'dir': 'eval_full',
                'name': 'FULL MedXplain-VQA',
                'description': 'Complete system with all components'
            }
        }
    
    def _load_semantic_model_safe(self):
        """Load semantic model với compatibility fallbacks"""
        try:
            import torch
            if not hasattr(torch, 'get_default_device'):
                torch.get_default_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded successfully")
            return model
            
        except Exception as e:
            print(f"⚠️ Could not load sentence-transformers: {e}")
            print("📝 Using fallback similarity methods")
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
        """Medical domain appropriate relevance scoring"""
        if not predicted.strip() or not ground_truth.strip():
            return 0.0
        
        pred_lower = predicted.lower()
        gt_lower = ground_truth.lower()
        
        # Exact match bonus
        if pred_lower == gt_lower:
            return 1.0
        
        # Substring containment
        if gt_lower in pred_lower or pred_lower in gt_lower:
            return 0.8
        
        # Medical keywords overlap
        medical_keywords = [
            'cell', 'tissue', 'lesion', 'structure', 'gland', 'follicle',
            'tumor', 'carcinoma', 'melanoma', 'nevus', 'inflammation',
            'dermatitis', 'fibrosis', 'hyperplasia', 'dysplasia',
            'benign', 'malignant', 'pathology', 'diagnosis', 'demodex',
            'folliculorum', 'sebaceous', 'keratin', 'epidermis', 'dermis'
        ]
        
        pred_keywords = [kw for kw in medical_keywords if kw in pred_lower]
        gt_keywords = [kw for kw in medical_keywords if kw in gt_lower]
        
        if pred_keywords and gt_keywords:
            keyword_overlap = len(set(pred_keywords).intersection(set(gt_keywords)))
            keyword_union = len(set(pred_keywords).union(set(gt_keywords)))
            if keyword_union > 0:
                return 0.3 + 0.4 * (keyword_overlap / keyword_union)
        
        # Word-level Jaccard similarity
        pred_words = set(pred_lower.split())
        gt_words = set(gt_lower.split())
        
        intersection = len(pred_words.intersection(gt_words))
        union = len(pred_words.union(gt_words))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        return min(0.7, jaccard_score * 0.8)
    
    def compute_semantic_similarity(self, predicted, ground_truth):
        """Enhanced semantic similarity"""
        if self.semantic_model:
            try:
                pred_emb = self.semantic_model.encode([predicted.lower()])
                gt_emb = self.semantic_model.encode([ground_truth.lower()])
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
                return float(similarity)
                
            except Exception as e:
                print(f"Semantic similarity error: {e}")
        
        # Fallback to relevance score
        return self.compute_answer_relevance_score(predicted, ground_truth)
    
    def extract_medical_concepts(self, text):
        """Extract medical concepts từ text"""
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
        
        return 0.1
    
    def analyze_mode_performance(self, mode_results):
        """Comprehensive performance analysis"""
        if not mode_results:
            return {}
        
        total_samples = len(mode_results)
        successful_samples = sum(1 for r in mode_results if r.get('success', False))
        success_rate = successful_samples / total_samples if total_samples > 0 else 0
        
        # Compute metrics for successful samples
        answer_relevance_scores = []
        semantic_similarity_scores = []
        clinical_accuracy_scores = []
        
        for result in mode_results:
            if result.get('success', False):
                predicted = result.get('unified_answer', '')
                ground_truth = result.get('ground_truth', '')
                
                if predicted and ground_truth:
                    # Answer relevance
                    relevance_score = self.compute_answer_relevance_score(predicted, ground_truth)
                    answer_relevance_scores.append(relevance_score)
                    
                    # Semantic similarity
                    semantic_score = self.compute_semantic_similarity(predicted, ground_truth)
                    semantic_similarity_scores.append(semantic_score)
                    
                    # Clinical accuracy
                    clinical_score = self.compute_clinical_accuracy_score(predicted, ground_truth)
                    clinical_accuracy_scores.append(clinical_score)
        
        # Query reformulation quality
        reformulation_qualities = [r['reformulation_quality'] for r in mode_results if 'reformulation_quality' in r]
        
        # Attention analysis
        attention_metrics = self.analyze_attention_quality(mode_results)
        
        # Reasoning analysis
        reasoning_metrics = self.analyze_reasoning_quality(mode_results)
        
        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': success_rate,
            'answer_relevance': {
                'mean': np.mean(answer_relevance_scores) if answer_relevance_scores else 0,
                'std': np.std(answer_relevance_scores) if answer_relevance_scores else 0,
                'median': np.median(answer_relevance_scores) if answer_relevance_scores else 0,
                'scores': answer_relevance_scores
            },
            'semantic_similarity': {
                'mean': np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0,
                'std': np.std(semantic_similarity_scores) if semantic_similarity_scores else 0,
                'median': np.median(semantic_similarity_scores) if semantic_similarity_scores else 0,
                'scores': semantic_similarity_scores
            },
            'clinical_accuracy': {
                'mean': np.mean(clinical_accuracy_scores) if clinical_accuracy_scores else 0,
                'std': np.std(clinical_accuracy_scores) if clinical_accuracy_scores else 0,
                'median': np.median(clinical_accuracy_scores) if clinical_accuracy_scores else 0,
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
        """Analyze attention quality"""
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
        """Analyze reasoning quality"""
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
    
    def create_ablation_table(self, all_analysis):
        """Create ablation study table"""
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        table_data = []
        
        for mode_key in ablation_order:
            if mode_key not in all_analysis:
                continue
                
            analysis = all_analysis[mode_key]
            config = self.mode_configs[mode_key]
            
            row = {
                'Method': config['name'],
                'Success Rate (%)': f"{analysis['success_rate']*100:.1f}",
                'Answer Relevance': f"{analysis['answer_relevance']['mean']:.3f} ± {analysis['answer_relevance']['std']:.3f}",
                'Semantic Similarity': f"{analysis['semantic_similarity']['mean']:.3f} ± {analysis['semantic_similarity']['std']:.3f}",
                'Clinical Accuracy': f"{analysis['clinical_accuracy']['mean']:.3f} ± {analysis['clinical_accuracy']['std']:.3f}",
                'Query Quality': f"{analysis['reformulation_quality']['mean']:.3f}" if analysis['reformulation_quality']['count'] > 0 else "N/A",
                'Attention Score': f"{analysis['attention_metrics']['avg_attention_score']:.3f}" if analysis['attention_metrics']['total_images_with_bbox'] > 0 else "N/A",
                'Reasoning Conf.': f"{analysis['reasoning_metrics']['avg_reasoning_confidence']:.3f}" if analysis['reasoning_metrics']['total_with_reasoning'] > 0 else "N/A"
            }
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def create_performance_plots(self, all_analysis, output_dir):
        """Create performance plots"""
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        modes = []
        relevance_scores = []
        semantic_scores = []
        clinical_scores = []
        
        for mode_key in ablation_order:
            if mode_key in all_analysis:
                analysis = all_analysis[mode_key]
                config = self.mode_configs[mode_key]
                
                modes.append(config['name'].replace(' + ', '\n+ '))
                relevance_scores.append(analysis['answer_relevance']['mean'])
                semantic_scores.append(analysis['semantic_similarity']['mean'])
                clinical_scores.append(analysis['clinical_accuracy']['mean'])
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Answer Relevance
        bars1 = ax1.bar(range(len(modes)), relevance_scores, color='lightblue', alpha=0.8)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Answer Relevance Score')
        ax1.set_title('Answer Relevance Comparison')
        ax1.set_xticks(range(len(modes)))
        ax1.set_xticklabels(modes, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars1, relevance_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Semantic Similarity
        bars2 = ax2.bar(range(len(modes)), semantic_scores, color='lightgreen', alpha=0.8)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Semantic Similarity Score') 
        ax2.set_title('Semantic Similarity Comparison')
        ax2.set_xticks(range(len(modes)))
        ax2.set_xticklabels(modes, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars2, semantic_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Clinical Accuracy
        bars3 = ax3.bar(range(len(modes)), clinical_scores, color='lightcoral', alpha=0.8)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Clinical Accuracy Score')
        ax3.set_title('Clinical Accuracy Comparison')
        ax3.set_xticks(range(len(modes)))
        ax3.set_xticklabels(modes, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars3, clinical_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Combined Performance
        combined_scores = [(r + s + c) / 3 for r, s, c in zip(relevance_scores, semantic_scores, clinical_scores)]
        bars4 = ax4.bar(range(len(modes)), combined_scores, color='gold', alpha=0.8)
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Combined Performance Score')
        ax4.set_title('Overall Performance Comparison')
        ax4.set_xticks(range(len(modes)))
        ax4.set_xticklabels(modes, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars4, combined_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_plots.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance plots saved")
    
    def run_analysis(self, output_dir="data/paper_results"):
        """Run complete analysis"""
        print("🚀 Starting MedXplain-VQA evaluation analysis...")
        print("="*60)
        
        # Load all results
        all_results = self.load_all_results()
        
        if not all_results:
            print("❌ No results found.")
            return
        
        # Analyze each mode
        print("\n📊 Analyzing performance by mode...")
        all_analysis = {}
        
        for mode_key, mode_results in all_results.items():
            config = self.mode_configs[mode_key]
            print(f"\n🔍 Analyzing {config['name']}...")
            
            analysis = self.analyze_mode_performance(mode_results)
            all_analysis[mode_key] = analysis
            
            # Print summary
            print(f"  • Samples: {analysis['total_samples']} total, {analysis['successful_samples']} successful")
            print(f"  • Success Rate: {analysis['success_rate']*100:.1f}%")
            print(f"  • Answer Relevance: {analysis['answer_relevance']['mean']:.3f} ± {analysis['answer_relevance']['std']:.3f}")
            print(f"  • Semantic Similarity: {analysis['semantic_similarity']['mean']:.3f} ± {analysis['semantic_similarity']['std']:.3f}")
            print(f"  • Clinical Accuracy: {analysis['clinical_accuracy']['mean']:.3f} ± {analysis['clinical_accuracy']['std']:.3f}")
            
            if analysis['attention_metrics']['total_images_with_bbox'] > 0:
                print(f"  • Attention Quality: {analysis['attention_metrics']['avg_attention_score']:.3f}")
            
            if analysis['reasoning_metrics']['total_with_reasoning'] > 0:
                print(f"  • Reasoning Confidence: {analysis['reasoning_metrics']['avg_reasoning_confidence']:.3f}")
        
        # Create tables and plots
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📋 Creating ablation study table...")
        ablation_df = self.create_ablation_table(all_analysis)
        
        # Generate LaTeX table
        ablation_latex = ablation_df.to_latex(
            index=False,
            escape=False,
            caption="Ablation Study: MedXplain-VQA Component Performance Analysis",
            label="tab:medxplain_ablation",
            column_format="l|c|c|c|c|c|c|c"
        )
        
        latex_file = os.path.join(output_dir, "ablation_table.tex")
        with open(latex_file, 'w') as f:
            f.write(ablation_latex)
        
        # Create plots
        self.create_performance_plots(all_analysis, output_dir)
        
        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_analysis, f, indent=2, default=self._json_serialize)
        
        print(f"\n🎉 Analysis complete! Results saved to {output_dir}")
        print(f"📊 Files generated:")
        print(f"  • ablation_table.tex (LaTeX table)")
        print(f"  • performance_plots.png (Visualization)")
        print(f"  • evaluation_results.json (Full data)")
        
        # Print final summary
        best_clinical = max(all_analysis.values(), key=lambda x: x['clinical_accuracy']['mean'])
        best_mode = [k for k, v in all_analysis.items() if v == best_clinical][0]
        best_config = self.mode_configs[best_mode]
        
        print(f"\n📈 FINAL SUMMARY:")
        print(f"🏆 Best Clinical Accuracy: {best_config['name']}")
        print(f"   Clinical Accuracy: {best_clinical['clinical_accuracy']['mean']:.3f}")
        print(f"   Answer Relevance: {best_clinical['answer_relevance']['mean']:.3f}")
        print(f"   Success Rate: {best_clinical['success_rate']*100:.1f}%")
        
        print(f"\n📋 ABLATION STUDY TABLE:")
        print(ablation_df.to_string(index=False))
        
        return all_analysis, ablation_df
    
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
    print("🎯 MedXplain-VQA Evaluation Results Analyzer")
    print("="*50)
    
    analyzer = EvaluationAnalyzer(base_data_dir="data")
    
    try:
        all_analysis, ablation_df = analyzer.run_analysis()
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
