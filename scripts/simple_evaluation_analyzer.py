#!/usr/bin/env python
"""
🎯 SIMPLE EVALUATION ANALYZER: Fast analysis without sentence-transformers
Day 2: Quick analysis for paper preparation (3-day timeline)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re

class SimpleEvaluationAnalyzer:
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = base_data_dir
        
        # Define mode configurations
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
    
    def compute_word_overlap_similarity(self, predicted, ground_truth):
        """Simple word overlap similarity thay vì semantic embedding"""
        if not predicted or not ground_truth:
            return 0.0
            
        # Clean and tokenize
        pred_words = set(re.findall(r'\b\w+\b', predicted.lower()))
        gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Jaccard similarity
        intersection = len(pred_words.intersection(gt_words))
        union = len(pred_words.union(gt_words))
        
        if union == 0:
            return 1.0
            
        jaccard = intersection / union
        
        # Also compute recall-based similarity (more relevant cho medical terms)
        recall = intersection / len(gt_words)
        
        # Combined score (weighted average)
        combined_score = 0.3 * jaccard + 0.7 * recall
        
        return combined_score
    
    def extract_medical_terms(self, text):
        """Extract medical terms and entities"""
        if not text:
            return []
            
        # Medical term patterns
        medical_patterns = [
            r'\b\w*oma\b',      # tumors: melanoma, carcinoma, adenoma
            r'\b\w*itis\b',     # inflammations: dermatitis, arthritis
            r'\b\w*osis\b',     # conditions: fibrosis, necrosis
            r'\b\w*pathy\b',    # diseases: neuropathy, cardiomyopathy
            r'\bcell[s]?\b',
            r'\btissue[s]?\b',
            r'\blesion[s]?\b',
            r'\bstructure[s]?\b',
            r'\bgland[s]?\b',
            r'\bfollicle[s]?\b',
            r'\bnucleus\b',
            r'\bnuclei\b',
            r'\bcytoplasm\b',
            r'\bepithelium\b',
            r'\bepithelial\b',
            r'\bstroma\b',
            r'\binfiltrate\b',
            r'\binfiltration\b',
            r'\bhyperplasia\b',
            r'\bdysplasia\b',
            r'\bmetaplasia\b',
            r'\bneoplasm\b',
            r'\bcarcinoma\b',
            r'\badenocarcinoma\b',
            r'\bmelanoma\b',
            r'\bnevus\b',
            r'\bpapilloma\b',
            r'\bfibroma\b',
            r'\blipoma\b',
            r'\bsarcoma\b'
        ]
        
        medical_terms = []
        text_lower = text.lower()
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            medical_terms.extend(matches)
        
        return list(set(medical_terms))  # Remove duplicates
    
    def compute_medical_term_overlap(self, predicted, ground_truth):
        """Compute overlap của medical terms specifically"""
        pred_terms = set(self.extract_medical_terms(predicted))
        gt_terms = set(self.extract_medical_terms(ground_truth))
        
        if len(gt_terms) == 0:
            return 1.0 if len(pred_terms) == 0 else 0.0
        
        intersection = len(pred_terms.intersection(gt_terms))
        recall = intersection / len(gt_terms)
        
        # If no medical terms in ground truth, use general word overlap
        if len(gt_terms) == 0:
            return self.compute_word_overlap_similarity(predicted, ground_truth)
        
        return recall
    
    def compute_length_normalized_score(self, predicted, ground_truth):
        """Penalize excessively long answers"""
        if not predicted or not ground_truth:
            return 0.0
            
        pred_len = len(predicted.split())
        gt_len = len(ground_truth.split())
        
        if gt_len == 0:
            return 1.0 if pred_len == 0 else 0.0
        
        # Length ratio penalty
        length_ratio = pred_len / gt_len
        
        if length_ratio <= 1.0:
            length_penalty = 1.0
        elif length_ratio <= 2.0:
            length_penalty = 1.0 - 0.1 * (length_ratio - 1.0)  # Small penalty
        elif length_ratio <= 5.0:
            length_penalty = 0.9 - 0.2 * (length_ratio - 2.0)  # Medium penalty
        else:
            length_penalty = 0.3  # Large penalty for very long answers
        
        # Base similarity
        base_similarity = self.compute_word_overlap_similarity(predicted, ground_truth)
        
        return base_similarity * length_penalty
    
    def analyze_mode_performance(self, mode_results):
        """Analyze performance của một mode"""
        if not mode_results:
            return {}
        
        # Basic metrics
        total_samples = len(mode_results)
        successful_samples = sum(1 for r in mode_results if r.get('success', False))
        success_rate = successful_samples / total_samples if total_samples > 0 else 0
        
        # Similarity scores
        word_overlap_scores = []
        medical_term_scores = []
        length_normalized_scores = []
        
        for result in mode_results:
            if result.get('success', False):
                predicted = result.get('unified_answer', '')
                ground_truth = result.get('ground_truth', '')
                
                if predicted and ground_truth:
                    # Word overlap similarity
                    word_score = self.compute_word_overlap_similarity(predicted, ground_truth)
                    word_overlap_scores.append(word_score)
                    
                    # Medical term overlap
                    medical_score = self.compute_medical_term_overlap(predicted, ground_truth)
                    medical_term_scores.append(medical_score)
                    
                    # Length normalized score
                    length_score = self.compute_length_normalized_score(predicted, ground_truth)
                    length_normalized_scores.append(length_score)
        
        # Query reformulation quality
        reformulation_qualities = []
        for result in mode_results:
            if 'reformulation_quality' in result and result['reformulation_quality'] is not None:
                reformulation_qualities.append(result['reformulation_quality'])
        
        # Attention analysis
        attention_metrics = self.analyze_attention_quality(mode_results)
        
        # Chain-of-thought analysis
        reasoning_metrics = self.analyze_reasoning_quality(mode_results)
        
        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': success_rate,
            'word_overlap_similarity': {
                'mean': np.mean(word_overlap_scores) if word_overlap_scores else 0,
                'std': np.std(word_overlap_scores) if word_overlap_scores else 0,
                'scores': word_overlap_scores
            },
            'medical_term_overlap': {
                'mean': np.mean(medical_term_scores) if medical_term_scores else 0,
                'std': np.std(medical_term_scores) if medical_term_scores else 0,
                'scores': medical_term_scores
            },
            'length_normalized_score': {
                'mean': np.mean(length_normalized_scores) if length_normalized_scores else 0,
                'std': np.std(length_normalized_scores) if length_normalized_scores else 0,
                'scores': length_normalized_scores
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
        
        # Count reasoning flow types
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
    
    def create_ablation_study_table(self, all_analysis):
        """Create ablation study comparison table"""
        
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
                'Word Overlap': f"{analysis['word_overlap_similarity']['mean']:.3f} ± {analysis['word_overlap_similarity']['std']:.3f}",
                'Medical Terms': f"{analysis['medical_term_overlap']['mean']:.3f} ± {analysis['medical_term_overlap']['std']:.3f}",
                'Length Normalized': f"{analysis['length_normalized_score']['mean']:.3f} ± {analysis['length_normalized_score']['std']:.3f}",
                'Query Quality': f"{analysis['reformulation_quality']['mean']:.3f}" if analysis['reformulation_quality']['count'] > 0 else "N/A",
                'Attention Regions': f"{analysis['attention_metrics']['avg_regions_per_image']:.1f}" if analysis['attention_metrics']['total_images_with_bbox'] > 0 else "N/A",
                'Reasoning Conf.': f"{analysis['reasoning_metrics']['avg_reasoning_confidence']:.3f}" if analysis['reasoning_metrics']['total_with_reasoning'] > 0 else "N/A"
            }
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def create_performance_plots(self, all_analysis, output_dir):
        """Create performance comparison plots"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        ablation_order = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
        
        modes = []
        word_overlap_scores = []
        medical_term_scores = []
        success_rates = []
        
        for mode_key in ablation_order:
            if mode_key in all_analysis:
                analysis = all_analysis[mode_key]
                config = self.mode_configs[mode_key]
                
                modes.append(config['name'])
                word_overlap_scores.append(analysis['word_overlap_similarity']['mean'])
                medical_term_scores.append(analysis['medical_term_overlap']['mean'])
                success_rates.append(analysis['success_rate'] * 100)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Word Overlap Similarity
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(modes)), word_overlap_scores, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Word Overlap Similarity')
        ax1.set_title('Word Overlap Similarity Comparison')
        ax1.set_xticks(range(len(modes)))
        ax1.set_xticklabels([m.replace(' + ', '\n+ ') for m in modes], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars1, word_overlap_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Medical Term Overlap
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(modes)), medical_term_scores, color='lightgreen', alpha=0.8)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Medical Term Overlap')
        ax2.set_title('Medical Term Overlap Comparison')
        ax2.set_xticks(range(len(modes)))
        ax2.set_xticklabels([m.replace(' + ', '\n+ ') for m in modes], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars2, medical_term_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Success Rate
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(modes)), success_rates, color='coral', alpha=0.8)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Processing Success Rate')
        ax3.set_xticks(range(len(modes)))
        ax3.set_xticklabels([m.replace(' + ', '\n+ ') for m in modes], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 105)
        
        for bar, rate in zip(bars3, success_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Combined Score Comparison  
        ax4 = axes[1, 1]
        combined_scores = [(w + m) / 2 for w, m in zip(word_overlap_scores, medical_term_scores)]
        bars4 = ax4.bar(range(len(modes)), combined_scores, color='gold', alpha=0.8)
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Combined Score')
        ax4.set_title('Combined Performance Score')
        ax4.set_xticks(range(len(modes)))
        ax4.set_xticklabels([m.replace(' + ', '\n+ ') for m in modes], rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars4, combined_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance plots saved to {output_dir}")
    
    def generate_latex_table(self, ablation_df, output_dir):
        """Generate LaTeX table"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        latex_table = ablation_df.to_latex(
            index=False,
            escape=False,
            caption="MedXplain-VQA Ablation Study: Component-wise Performance Analysis",
            label="tab:medxplain_ablation",
            column_format="l|c|c|c|c|c|c|c"
        )
        
        # Add some formatting
        latex_table = latex_table.replace('\\toprule', '\\hline')
        latex_table = latex_table.replace('\\midrule', '\\hline')
        latex_table = latex_table.replace('\\bottomrule', '\\hline')
        
        table_file = os.path.join(output_dir, "medxplain_ablation_table.tex")
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        print(f"✅ LaTeX table saved to {table_file}")
        return latex_table
    
    def run_complete_analysis(self, output_dir="data/paper_results"):
        """Run complete analysis"""
        
        print("🚀 Starting MedXplain-VQA Evaluation Analysis...")
        print("="*60)
        
        # Load all results
        all_results = self.load_all_results()
        
        if not all_results:
            print("❌ No results found. Make sure evaluation data exists.")
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
            print(f"  • Word Overlap: {analysis['word_overlap_similarity']['mean']:.3f} ± {analysis['word_overlap_similarity']['std']:.3f}")
            print(f"  • Medical Terms: {analysis['medical_term_overlap']['mean']:.3f} ± {analysis['medical_term_overlap']['std']:.3f}")
            
            if analysis['attention_metrics']['total_images_with_bbox'] > 0:
                print(f"  • Attention Regions: {analysis['attention_metrics']['avg_regions_per_image']:.1f}")
            
            if analysis['reasoning_metrics']['total_with_reasoning'] > 0:
                print(f"  • Reasoning Confidence: {analysis['reasoning_metrics']['avg_reasoning_confidence']:.3f}")
        
        # Create ablation table
        print(f"\n📋 Creating ablation study table...")
        ablation_df = self.create_ablation_study_table(all_analysis)
        
        # Generate LaTeX table
        print(f"\n📄 Generating LaTeX table...")
        latex_table = self.generate_latex_table(ablation_df, output_dir)
        
        # Create plots
        print(f"\n📈 Creating performance plots...")
        self.create_performance_plots(all_analysis, output_dir)
        
        # Save summary
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_analysis, f, indent=2, default=str)
        
        print(f"\n🎉 Analysis complete! Results saved to {output_dir}")
        
        # Print final summary
        best_combined = None
        best_mode = None
        best_score = 0
        
        for mode_key, analysis in all_analysis.items():
            word_score = analysis['word_overlap_similarity']['mean']
            medical_score = analysis['medical_term_overlap']['mean']
            combined_score = (word_score + medical_score) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_mode = mode_key
                best_combined = analysis
        
        if best_mode:
            config = self.mode_configs[best_mode]
            print(f"\n🏆 BEST PERFORMING METHOD: {config['name']}")
            print(f"   Combined Score: {best_score:.3f}")
            print(f"   Word Overlap: {best_combined['word_overlap_similarity']['mean']:.3f}")
            print(f"   Medical Terms: {best_combined['medical_term_overlap']['mean']:.3f}")
            print(f"   Success Rate: {best_combined['success_rate']*100:.1f}%")
        
        print(f"\n📊 ABLATION STUDY TABLE:")
        print(ablation_df.to_string(index=False))
        
        return all_analysis, ablation_df

def main():
    print("🎯 Simple MedXplain-VQA Evaluation Analyzer")
    print("="*50)
    
    analyzer = SimpleEvaluationAnalyzer(base_data_dir="data")
    
    try:
        all_analysis, ablation_df = analyzer.run_complete_analysis()
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
