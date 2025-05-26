#!/usr/bin/env python
"""
🎯 FIXED FINAL EVALUATION: Correct names + Performance analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Fix torch compatibility
import torch
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EvaluationFinalFixed:
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = base_data_dir
        
        # Load semantic model with compatibility fix
        self.semantic_model = self._load_semantic_model_safe()
        
        # 🔧 FIXED: Clean method names for paper
        self.mode_configs = {
            'basic': {
                'dir': 'eval_basic',
                'paper_name': 'Basic',
                'full_name': 'BLIP + Gemini'
            },
            'explainable': {
                'dir': 'eval_explainable', 
                'paper_name': 'Explainable',
                'full_name': 'BLIP + Query Reform + GradCAM'
            },
            'explainable_bbox': {
                'dir': 'eval_bbox',
                'paper_name': 'ExplainableBBox',
                'full_name': 'BLIP + ... + Bounding Boxes'
            },
            'enhanced': {
                'dir': 'eval_enhanced',
                'paper_name': 'Enhanced',
                'full_name': 'BLIP + ... + Chain-of-Thought'
            },
            'enhanced_bbox': {
                'dir': 'eval_full',
                'paper_name': 'MedXplain-VQA',
                'full_name': 'Complete System (All Components)'
            }
        }
    
    def _load_semantic_model_safe(self):
        """Load semantic model with compatibility fix"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded successfully")
            return model
        except Exception as e:
            print(f"⚠️ Using fallback similarity: {e}")
            return None
    
    def load_all_results(self):
        """Load evaluation results"""
        all_results = {}
        
        for mode_key, config in self.mode_configs.items():
            results_dir = os.path.join(self.base_data_dir, config['dir'])
            
            if not os.path.exists(results_dir):
                print(f"⚠️ Directory not found: {results_dir}")
                continue
                
            mode_results = []
            json_files = list(Path(results_dir).glob("*.json"))
            
            print(f"📂 Loading {len(json_files)} results from {config['paper_name']}...")
            
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
            print(f"✅ Loaded {len(mode_results)} results for {config['paper_name']}")
        
        return all_results
    
    def compute_answer_relevance_score(self, predicted, ground_truth):
        """Medical domain answer relevance"""
        if not predicted.strip() or not ground_truth.strip():
            return 0.0
        
        pred_lower = predicted.lower()
        gt_lower = ground_truth.lower()
        
        # Exact match
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
            'folliculorum', 'sebaceous', 'keratin', 'epithelial'
        ]
        
        pred_keywords = [kw for kw in medical_keywords if kw in pred_lower]
        gt_keywords = [kw for kw in medical_keywords if kw in gt_lower]
        
        if pred_keywords and gt_keywords:
            overlap = len(set(pred_keywords).intersection(set(gt_keywords)))
            union = len(set(pred_keywords).union(set(gt_keywords)))
            if union > 0:
                return 0.3 + 0.5 * (overlap / union)
        
        # Word overlap
        pred_words = set(pred_lower.split())
        gt_words = set(gt_lower.split())
        
        intersection = len(pred_words.intersection(gt_words))
        union = len(pred_words.union(gt_words))
        
        return (intersection / union * 0.7) if union > 0 else 0.0
    
    def compute_semantic_similarity(self, predicted, ground_truth):
        """Semantic similarity with fallback"""
        if self.semantic_model:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                
                pred_emb = self.semantic_model.encode([predicted.lower()])
                gt_emb = self.semantic_model.encode([ground_truth.lower()])
                
                similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
                return float(similarity)
                
            except Exception:
                pass
                
        return self.compute_answer_relevance_score(predicted, ground_truth)
    
    def analyze_mode_performance(self, mode_results):
        """Analyze performance with detailed breakdown"""
        if not mode_results:
            return {}
        
        total_samples = len(mode_results)
        successful_samples = sum(1 for r in mode_results if r.get('success', False))
        success_rate = successful_samples / total_samples if total_samples > 0 else 0
        
        # Compute scores
        answer_relevance_scores = []
        semantic_similarity_scores = []
        
        for result in mode_results:
            if result.get('success', False):
                predicted = result.get('unified_answer', '')
                ground_truth = result.get('ground_truth', '')
                
                if predicted and ground_truth:
                    relevance_score = self.compute_answer_relevance_score(predicted, ground_truth)
                    answer_relevance_scores.append(relevance_score)
                    
                    semantic_score = self.compute_semantic_similarity(predicted, ground_truth)
                    semantic_similarity_scores.append(semantic_score)
        
        # Component analysis
        reformulation_qualities = []
        attention_scores = []
        reasoning_confidences = []
        
        for result in mode_results:
            if 'reformulation_quality' in result:
                reformulation_qualities.append(result['reformulation_quality'])
            
            if result.get('bbox_regions_count', 0) > 0 and 'bounding_box_analysis' in result:
                attention_scores.append(result['bounding_box_analysis'].get('average_attention_score', 0))
            
            if 'reasoning_analysis' in result:
                reasoning_confidences.append(result['reasoning_analysis'].get('reasoning_confidence', 0))
        
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
            'reformulation_quality': {
                'mean': np.mean(reformulation_qualities) if reformulation_qualities else 0,
                'count': len(reformulation_qualities)
            },
            'attention_quality': {
                'mean': np.mean(attention_scores) if attention_scores else 0,
                'count': len(attention_scores)
            },
            'reasoning_confidence': {
                'mean': np.mean(reasoning_confidences) if reasoning_confidences else 0,
                'count': len(reasoning_confidences)
            }
        }
    
    def create_paper_table(self, all_analysis):
        """Create clean paper table"""
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        table_data = []
        
        for mode_key in ablation_order:
            if mode_key not in all_analysis:
                continue
                
            analysis = all_analysis[mode_key]
            config = self.mode_configs[mode_key]
            
            row = {
                'Method': config['paper_name'],
                'Success Rate': f"{analysis['success_rate']*100:.1f}%",
                'Answer Relevance': f"{analysis['answer_relevance']['mean']:.3f}",
                'Semantic Similarity': f"{analysis['semantic_similarity']['mean']:.3f}",
                'Query Quality': f"{analysis['reformulation_quality']['mean']:.3f}" if analysis['reformulation_quality']['count'] > 0 else "—",
                'Attention Score': f"{analysis['attention_quality']['mean']:.3f}" if analysis['attention_quality']['count'] > 0 else "—",
                'Reasoning Confidence': f"{analysis['reasoning_confidence']['mean']:.3f}" if analysis['reasoning_confidence']['count'] > 0 else "—"
            }
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def analyze_performance_issues(self, all_analysis):
        """🔍 Analyze why Full system performs worse"""
        print("\n🔍 PERFORMANCE ISSUE ANALYSIS:")
        print("="*50)
        
        # Compare key methods
        bbox_analysis = all_analysis.get('explainable_bbox', {})
        full_analysis = all_analysis.get('enhanced_bbox', {})
        
        if bbox_analysis and full_analysis:
            print(f"📊 BLIP + Bounding Boxes:")
            print(f"  Answer Relevance: {bbox_analysis['answer_relevance']['mean']:.3f}")
            print(f"  Semantic Similarity: {bbox_analysis['semantic_similarity']['mean']:.3f}")
            
            print(f"\n📊 Full MedXplain-VQA:")
            print(f"  Answer Relevance: {full_analysis['answer_relevance']['mean']:.3f}")
            print(f"  Semantic Similarity: {full_analysis['semantic_similarity']['mean']:.3f}")
            print(f"  Reasoning Confidence: {full_analysis['reasoning_confidence']['mean']:.3f}")
            
            # Performance comparison
            relevance_diff = full_analysis['answer_relevance']['mean'] - bbox_analysis['answer_relevance']['mean']
            semantic_diff = full_analysis['semantic_similarity']['mean'] - bbox_analysis['semantic_similarity']['mean']
            
            print(f"\n📈 PERFORMANCE DIFFERENCE:")
            print(f"  Answer Relevance: {relevance_diff:+.3f}")
            print(f"  Semantic Similarity: {semantic_diff:+.3f}")
            
            if relevance_diff < 0 or semantic_diff < 0:
                print(f"\n⚠️ ISSUE IDENTIFIED: Full system performs worse!")
                print(f"💡 POSSIBLE CAUSES:")
                print(f"  1. Chain-of-Thought adds complexity without improving accuracy")
                print(f"  2. Component interference between bbox attention and reasoning")
                print(f"  3. Small sample size (5) creates high variance")
                print(f"  4. CoT may over-explain, reducing direct answer quality")
                
                print(f"\n🎯 RECOMMENDATION:")
                print(f"  Use 'BLIP + Bounding Boxes' as best performing method in paper")
                print(f"  Scale to 50 samples to confirm trend")
                print(f"  Consider Full system as 'comprehensive' rather than 'best'")
    
    def create_fixed_plots(self, all_analysis, output_dir):
        """🔧 FIXED: Clean visualization with proper names"""
        
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        # 🔧 FIXED: Use paper_name instead of full_name
        methods = []
        relevance_scores = []
        semantic_scores = []
        
        for mode_key in ablation_order:
            if mode_key in all_analysis:
                analysis = all_analysis[mode_key]
                config = self.mode_configs[mode_key]
                
                methods.append(config['paper_name'])  # 🔧 FIXED: Clean names
                relevance_scores.append(analysis['answer_relevance']['mean'])
                semantic_scores.append(analysis['semantic_similarity']['mean'])
        
        # Create clean visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x_pos = np.arange(len(methods))
        
        # Plot 1: Answer Relevance
        bars1 = ax1.bar(x_pos, relevance_scores, color='skyblue', alpha=0.8, edgecolor='navy')
        ax1.set_xlabel('Method', fontsize=12)
        ax1.set_ylabel('Answer Relevance Score', fontsize=12)
        ax1.set_title('Answer Relevance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods, rotation=0)  # 🔧 FIXED: No rotation needed
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars1, relevance_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Semantic Similarity
        bars2 = ax2.bar(x_pos, semantic_scores, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Semantic Similarity Score', fontsize=12)
        ax2.set_title('Semantic Similarity Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods, rotation=0)  # 🔧 FIXED: No rotation needed
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars2, semantic_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "paper_evaluation_fixed.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Fixed visualization saved")
    
    def run_complete_analysis(self, output_dir="data/paper_results_final"):
        """Run complete analysis with issue detection"""
        print("🚀 Final Paper Evaluation (Fixed)")
        print("="*40)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        all_results = self.load_all_results()
        
        if not all_results:
            print("❌ No results found")
            return
        
        # Analyze performance
        print("\n📊 Analyzing performance...")
        all_analysis = {}
        
        for mode_key, mode_results in all_results.items():
            config = self.mode_configs[mode_key]
            print(f"\n🔍 {config['paper_name']} ({len(mode_results)} samples)")
            
            analysis = self.analyze_mode_performance(mode_results)
            all_analysis[mode_key] = analysis
            
            print(f"  Success Rate: {analysis['success_rate']*100:.1f}%")
            print(f"  Answer Relevance: {analysis['answer_relevance']['mean']:.3f}")
            print(f"  Semantic Similarity: {analysis['semantic_similarity']['mean']:.3f}")
        
        # Analyze performance issues
        self.analyze_performance_issues(all_analysis)
        
        # Create paper table
        print(f"\n📋 Creating paper table...")
        paper_df = self.create_paper_table(all_analysis)
        
        # Generate LaTeX
        latex_table = paper_df.to_latex(
            index=False,
            escape=False,
            caption="MedXplain-VQA Ablation Study Results",
            label="tab:medxplain_ablation",
            column_format="l|c|c|c|c|c|c"
        )
        
        # Save files
        latex_file = os.path.join(output_dir, "paper_table_fixed.tex")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        # Create fixed plots
        self.create_fixed_plots(all_analysis, output_dir)
        
        # Save complete results
        results_file = os.path.join(output_dir, "evaluation_results_fixed.json")
        with open(results_file, 'w') as f:
            json.dump(all_analysis, f, indent=2, default=self._json_serialize)
        
        print(f"\n🎉 Fixed evaluation completed!")
        print(f"📊 Files saved to: {output_dir}")
        print(f"  • paper_table_fixed.tex")
        print(f"  • paper_evaluation_fixed.png")
        print(f"  • evaluation_results_fixed.json")
        
        # Print final table
        print(f"\n📋 FINAL PAPER TABLE:")
        print(paper_df.to_string(index=False))
        
        # Best method recommendation
        best_semantic = max(all_analysis.values(), key=lambda x: x['semantic_similarity']['mean'])
        best_mode = [k for k, v in all_analysis.items() if v == best_semantic][0]
        best_config = self.mode_configs[best_mode]
        
        print(f"\n🏆 BEST PERFORMING METHOD FOR PAPER:")
        print(f"Method: {best_config['paper_name']}")
        print(f"Semantic Similarity: {best_semantic['semantic_similarity']['mean']:.3f}")
        print(f"Success Rate: {best_semantic['success_rate']*100:.1f}%")
        
        return all_analysis, paper_df
    
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
    evaluator = EvaluationFinalFixed()
    all_analysis, paper_df = evaluator.run_complete_analysis()

if __name__ == "__main__":
    main()
