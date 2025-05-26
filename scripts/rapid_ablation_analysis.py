#!/usr/bin/env python
"""
🔬 RAPID ABLATION ANALYSIS FOR MEDXPLAIN-VQA
============================================

Analyze and compare 5 different processing modes:
1. Basic (BLIP + Gemini)
2. Explainable (+ Query Reformulation + Grad-CAM)  
3. Explainable + BBox (+ Bounding Box extraction)
4. Enhanced (+ Chain-of-Thought reasoning)
5. Enhanced + BBox (Complete MedXplain-VQA system)

Author: MedXplain-VQA Team
Version: 1.0 - Rapid Comparison Framework
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

class ModeComparator:
    """🎯 Compare different MedXplain-VQA processing modes"""
    
    def __init__(self):
        self.mode_definitions = {
            'basic': {
                'name': 'Basic VQA',
                'description': 'BLIP + Gemini only',
                'components': ['BLIP inference', 'Gemini enhancement']
            },
            'explainable': {
                'name': 'Explainable VQA', 
                'description': 'BLIP + Query Reformulation + Grad-CAM',
                'components': ['BLIP inference', 'Query reformulation', 'Grad-CAM attention', 'Gemini enhancement']
            },
            'explainable_bbox': {
                'name': 'Explainable + BBox',
                'description': 'Explainable + Bounding Box extraction', 
                'components': ['BLIP inference', 'Query reformulation', 'Enhanced Grad-CAM', 'Bounding boxes', 'Gemini enhancement']
            },
            'enhanced': {
                'name': 'Enhanced VQA',
                'description': 'Explainable + Chain-of-Thought reasoning',
                'components': ['BLIP inference', 'Query reformulation', 'Grad-CAM attention', 'Chain-of-Thought', 'Gemini enhancement']
            },
            'enhanced_bbox': {
                'name': 'Complete MedXplain-VQA',
                'description': 'All components + Bounding Box extraction',
                'components': ['BLIP inference', 'Query reformulation', 'Enhanced Grad-CAM', 'Bounding boxes', 'Chain-of-Thought', 'Gemini enhancement']
            }
        }
    
    def load_mode_results(self, results_dirs: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Load results from different mode directories"""
        mode_results = {}
        
        for mode_name, results_dir in results_dirs.items():
            print(f"📁 Loading {mode_name} results from {results_dir}")
            
            results_path = Path(results_dir)
            if not results_path.exists():
                print(f"⚠️ Directory not found: {results_dir}")
                continue
            
            json_files = list(results_path.glob("*.json"))
            results = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results.append(data)
                except Exception as e:
                    print(f"⚠️ Error loading {json_file}: {e}")
                    continue
            
            mode_results[mode_name] = results
            print(f"✅ Loaded {len(results)} results for {mode_name}")
        
        return mode_results
    
    def analyze_mode_performance(self, mode_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze performance across different modes"""
        analysis = {
            'mode_comparison': {},
            'component_contribution': {},
            'performance_metrics': {}
        }
        
        # Import the fixed evaluator
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from medical_evaluation_suite import FixedMedicalEvaluator
        
        evaluator = FixedMedicalEvaluator(debug=False)
        
        # Analyze each mode
        for mode_name, results in mode_results.items():
            print(f"🔬 Analyzing {mode_name} mode ({len(results)} samples)...")
            
            if not results:
                continue
            
            # Evaluate all samples in this mode
            mode_evaluations = []
            for result in results:
                try:
                    evaluation = evaluator.evaluate_comprehensive_fixed(result)
                    mode_evaluations.append(evaluation)
                except Exception as e:
                    print(f"⚠️ Error evaluating sample in {mode_name}: {e}")
                    continue
            
            if not mode_evaluations:
                continue
            
            # Calculate mode statistics
            mode_stats = self._calculate_mode_statistics(mode_evaluations)
            analysis['mode_comparison'][mode_name] = mode_stats
            
            # Extract key metrics for comparison
            analysis['performance_metrics'][mode_name] = {
                'enhanced_medical_similarity': mode_stats['enhanced_medical_similarity']['mean'],
                'terminology_sophistication': mode_stats['terminology_sophistication']['mean'],
                'reasoning_confidence': mode_stats['explainability']['reasoning_confidence']['mean'],
                'attention_quality': mode_stats['explainability']['attention_quality']['mean'],
                'reformulation_quality': mode_stats['explainability']['reformulation_quality']['mean'],
                'overall_score': mode_stats['overall_medxplain_score']['mean'],
                'sample_count': len(mode_evaluations)
            }
        
        # Calculate component contributions
        analysis['component_contribution'] = self._calculate_component_contributions(
            analysis['performance_metrics']
        )
        
        return analysis
    
    def _calculate_mode_statistics(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a mode"""
        stats = {}
        
        # Basic metrics
        metrics = ['enhanced_medical_similarity', 'terminology_sophistication', 'overall_medxplain_score']
        
        for metric in metrics:
            values = [e.get(metric, 0.0) for e in evaluations if e.get(metric) is not None]
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Explainability metrics
        reasoning_scores = [e.get('explainability_metrics', {}).get('reasoning_confidence', 0.0) for e in evaluations]
        attention_scores = [e.get('explainability_metrics', {}).get('attention_avg_score', 0.0) for e in evaluations]  
        reformulation_scores = [e.get('explainability_metrics', {}).get('reformulation_quality', 0.0) for e in evaluations]
        
        stats['explainability'] = {
            'reasoning_confidence': self._stats_dict(reasoning_scores),
            'attention_quality': self._stats_dict(attention_scores),
            'reformulation_quality': self._stats_dict(reformulation_scores)
        }
        
        # Component usage
        component_scores = [e.get('component_completeness', 0.0) for e in evaluations]
        stats['component_integration'] = self._stats_dict(component_scores)
        
        return stats
    
    def _stats_dict(self, values: List[float]) -> Dict[str, float]:
        """Helper to calculate statistics dictionary"""
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    def _calculate_component_contributions(self, performance_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate individual component contributions"""
        contributions = {}
        
        # Define mode progression (each mode adds components)
        mode_progression = [
            'basic',
            'explainable', 
            'explainable_bbox',
            'enhanced',
            'enhanced_bbox'
        ]
        
        # Calculate incremental improvements
        baseline_score = performance_metrics.get('basic', {}).get('overall_score', 0.0)
        
        for i, mode in enumerate(mode_progression):
            if mode not in performance_metrics:
                continue
            
            mode_score = performance_metrics[mode]['overall_score']
            
            if i == 0:  # Basic mode
                contributions[mode] = {
                    'absolute_score': mode_score,
                    'improvement_over_previous': 0.0,
                    'improvement_over_baseline': 0.0,
                    'relative_improvement': 0.0
                }
            else:
                # Find previous available mode
                prev_score = baseline_score
                for j in range(i-1, -1, -1):
                    prev_mode = mode_progression[j]
                    if prev_mode in performance_metrics:
                        prev_score = performance_metrics[prev_mode]['overall_score']
                        break
                
                improvement_over_prev = mode_score - prev_score
                improvement_over_baseline = mode_score - baseline_score
                relative_improvement = (improvement_over_prev / prev_score * 100) if prev_score > 0 else 0
                
                contributions[mode] = {
                    'absolute_score': mode_score,
                    'improvement_over_previous': improvement_over_prev,
                    'improvement_over_baseline': improvement_over_baseline,
                    'relative_improvement': relative_improvement
                }
        
        return contributions
    
    def generate_comparison_report(self, analysis: Dict[str, Any], output_dir: str):
        """Generate comprehensive comparison report"""
        report_lines = []
        
        report_lines.append("🔬 MEDXPLAIN-VQA ABLATION STUDY RESULTS")
        report_lines.append("=" * 80)
        
        # Mode comparison summary
        report_lines.append("\n📊 MODE PERFORMANCE COMPARISON:")
        report_lines.append("-" * 50)
        
        performance_metrics = analysis.get('performance_metrics', {})
        
        # Create comparison table
        modes_data = []
        for mode_name, metrics in performance_metrics.items():
            mode_display = self.mode_definitions.get(mode_name, {}).get('name', mode_name)
            modes_data.append({
                'Mode': mode_display,
                'Overall Score': f"{metrics['overall_score']:.3f}",
                'Medical Similarity': f"{metrics['enhanced_medical_similarity']:.3f}",
                'Terminology': f"{metrics['terminology_sophistication']:.3f}",
                'Reasoning': f"{metrics['reasoning_confidence']:.3f}",
                'Attention': f"{metrics['attention_quality']:.3f}",
                'Samples': metrics['sample_count']
            })
        
        # Sort by overall score
        modes_data.sort(key=lambda x: float(x['Overall Score']), reverse=True)
        
        # Format table
        if modes_data:
            headers = list(modes_data[0].keys())
            col_widths = [max(len(str(row[col])) for row in [{'Mode': h} for h in headers] + modes_data) + 2 
                         for col in headers]
            
            # Header
            header_line = "|".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
            report_lines.append(header_line)
            report_lines.append("-" * len(header_line))
            
            # Data rows
            for row in modes_data:
                data_line = "|".join(f"{str(row[col]):^{w}}" for col, w in zip(headers, col_widths))
                report_lines.append(data_line)
        
        # Component contribution analysis
        report_lines.append("\n⚙️ COMPONENT CONTRIBUTION ANALYSIS:")
        report_lines.append("-" * 50)
        
        contributions = analysis.get('component_contribution', {})
        for mode_name, contrib in contributions.items():
            mode_display = self.mode_definitions.get(mode_name, {}).get('name', mode_name)
            
            report_lines.append(f"\n{mode_display}:")
            report_lines.append(f"  Absolute Score: {contrib['absolute_score']:.3f}")
            
            if contrib['improvement_over_previous'] != 0:
                report_lines.append(f"  Improvement over previous: +{contrib['improvement_over_previous']:.3f} "
                                  f"({contrib['relative_improvement']:+.1f}%)")
            
            if contrib['improvement_over_baseline'] != 0:
                report_lines.append(f"  Improvement over baseline: +{contrib['improvement_over_baseline']:.3f}")
        
        # Key insights
        report_lines.append("\n💡 KEY INSIGHTS:")
        report_lines.append("-" * 50)
        
        if performance_metrics:
            best_mode = max(performance_metrics.items(), key=lambda x: x[1]['overall_score'])
            best_mode_name = self.mode_definitions.get(best_mode[0], {}).get('name', best_mode[0])
            
            report_lines.append(f"🏆 Best performing mode: {best_mode_name} (score: {best_mode[1]['overall_score']:.3f})")
            
            # Identify strongest components
            strengths = []
            for mode_name, metrics in performance_metrics.items():
                if metrics['reasoning_confidence'] > 0.8:
                    strengths.append(f"Strong reasoning capabilities in {self.mode_definitions.get(mode_name, {}).get('name', mode_name)}")
                if metrics['attention_quality'] > 0.8:
                    strengths.append(f"High attention quality in {self.mode_definitions.get(mode_name, {}).get('name', mode_name)}")
                if metrics['reformulation_quality'] > 0.9:
                    strengths.append(f"Excellent query reformulation in {self.mode_definitions.get(mode_name, {}).get('name', mode_name)}")
            
            if strengths:
                report_lines.append("\n🎯 System Strengths:")
                for strength in strengths[:5]:  # Top 5 strengths
                    report_lines.append(f"  • {strength}")
        
        # Save report
        report_file = Path(output_dir) / "ablation_study_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Ablation study report saved to: {report_file}")
        
        # Also save detailed JSON
        json_file = Path(output_dir) / "ablation_study_detailed.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Detailed analysis saved to: {json_file}")
        
        return report_lines

def main():
    parser = argparse.ArgumentParser(description='🔬 Rapid Ablation Analysis for MedXplain-VQA')
    parser.add_argument('--basic-dir', type=str, default='data/eval_basic',
                       help='Directory with basic mode results')
    parser.add_argument('--explainable-dir', type=str, default='data/eval_explainable', 
                       help='Directory with explainable mode results')
    parser.add_argument('--bbox-dir', type=str, default='data/eval_bbox',
                       help='Directory with explainable + bbox results')
    parser.add_argument('--enhanced-dir', type=str, default='data/eval_enhanced',
                       help='Directory with enhanced mode results')
    parser.add_argument('--full-dir', type=str, default='data/eval_full',
                       help='Directory with complete system results')
    parser.add_argument('--output-dir', type=str, default='data/ablation_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('ablation_analysis', args.output_dir, level='INFO')
    
    logger.info("🚀 Starting Rapid Ablation Analysis")
    logger.info("🎯 Comparing 5 MedXplain-VQA processing modes")
    
    # Initialize comparator
    comparator = ModeComparator()
    
    # Load results from all modes
    results_dirs = {
        'basic': args.basic_dir,
        'explainable': args.explainable_dir,
        'explainable_bbox': args.bbox_dir,
        'enhanced': args.enhanced_dir,
        'enhanced_bbox': args.full_dir
    }
    
    logger.info("📂 Loading results from all modes...")
    mode_results = comparator.load_mode_results(results_dirs)
    
    if not mode_results:
        logger.error("❌ No valid results found!")
        return
    
    logger.info(f"✅ Loaded results for {len(mode_results)} modes")
    
    # Perform comparative analysis
    logger.info("🔬 Performing ablation analysis...")
    analysis = comparator.analyze_mode_performance(mode_results)
    
    # Generate comparison report
    logger.info("📊 Generating comparison report...")
    report_lines = comparator.generate_comparison_report(analysis, args.output_dir)
    
    # Print summary to console
    logger.info("\n" + "="*80)
    logger.info("🎯 ABLATION STUDY SUMMARY")
    logger.info("="*80)
    
    performance_metrics = analysis.get('performance_metrics', {})
    if performance_metrics:
        logger.info("\n📊 MODE PERFORMANCE RANKING:")
        
        # Sort modes by performance
        sorted_modes = sorted(performance_metrics.items(), 
                            key=lambda x: x[1]['overall_score'], reverse=True)
        
        for rank, (mode_name, metrics) in enumerate(sorted_modes, 1):
            mode_display = comparator.mode_definitions.get(mode_name, {}).get('name', mode_name)
            logger.info(f"  {rank}. {mode_display:<25} Score: {metrics['overall_score']:.3f} "
                       f"(Samples: {metrics['sample_count']})")
        
        # Component contribution summary
        contributions = analysis.get('component_contribution', {})
        if contributions:
            logger.info("\n⚙️ COMPONENT IMPROVEMENTS:")
            
            for mode_name, contrib in contributions.items():
                if contrib['improvement_over_previous'] > 0:
                    mode_display = comparator.mode_definitions.get(mode_name, {}).get('name', mode_name)
                    improvement = contrib['improvement_over_previous']
                    relative = contrib['relative_improvement']
                    logger.info(f"  {mode_display:<25} +{improvement:.3f} ({relative:+.1f}%)")
    
    logger.info(f"\n💾 Complete analysis saved to: {args.output_dir}")
    logger.info("🎉 Ablation analysis completed successfully!")

if __name__ == "__main__":
    main()
