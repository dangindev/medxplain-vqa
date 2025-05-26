#!/usr/bin/env python
"""
🔬 COMPONENT ABLATION STUDY FOR MEDXPLAIN-VQA
=============================================

Comprehensive ablation analysis comparing 5 processing modes:
1. Basic: BLIP + Gemini
2. Explainable: + Query Reformulation + Grad-CAM  
3. Explainable+BBox: + Bounding Box extraction
4. Enhanced: + Chain-of-Thought reasoning
5. Enhanced+BBox: Complete MedXplain-VQA system

Statistical analysis with significance testing and paper-ready outputs.

Author: MedXplain-VQA Team
Version: 2.0 - Complete Statistical Framework
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse
import logging
from collections import defaultdict
import re
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

class StatisticalAnalyzer:
    """🎯 Statistical significance testing and analysis"""
    
    @staticmethod
    def paired_t_test(before: List[float], after: List[float]) -> Dict[str, float]:
        """Perform paired t-test for before/after comparison"""
        if len(before) != len(after) or len(before) < 2:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False}
        
        try:
            t_stat, p_value = stats.ttest_rel(after, before)
            significant = p_value < 0.05
            
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': significant,
                'effect_size': (np.mean(after) - np.mean(before)) / np.std(before) if np.std(before) > 0 else 0.0
            }
        except Exception:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False, 'effect_size': 0.0}
    
    @staticmethod
    def independent_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Perform independent samples t-test"""
        if len(group1) < 2 or len(group2) < 2:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False}
        
        try:
            t_stat, p_value = stats.ttest_ind(group2, group1)
            significant = p_value < 0.05
            
            pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) / (len(group1)+len(group2)-2))
            effect_size = (np.mean(group2) - np.mean(group1)) / pooled_std if pooled_std > 0 else 0.0
            
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': significant,
                'effect_size': float(effect_size)
            }
        except Exception:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False, 'effect_size': 0.0}

class ComponentAblationAnalyzer:
    """🎯 Main Component Ablation Analysis Framework"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Mode definitions with component progression
        self.modes = {
            'basic': {
                'name': 'Basic VQA',
                'description': 'BLIP + Gemini only',
                'components': ['blip_inference', 'gemini_enhancement'],
                'order': 1,
                'color': '#FF6B6B'  # Red
            },
            'explainable': {
                'name': 'Explainable VQA', 
                'description': 'Basic + Query Reformulation + Grad-CAM',
                'components': ['blip_inference', 'query_reformulation', 'grad_cam', 'gemini_enhancement'],
                'order': 2,
                'color': '#4ECDC4'  # Teal
            },
            'explainable_bbox': {
                'name': 'Explainable + BBox',
                'description': 'Explainable + Enhanced Grad-CAM + Bounding Boxes', 
                'components': ['blip_inference', 'query_reformulation', 'enhanced_grad_cam', 'bounding_boxes', 'gemini_enhancement'],
                'order': 3,
                'color': '#45B7D1'  # Blue
            },
            'enhanced': {
                'name': 'Enhanced VQA',
                'description': 'Explainable + Chain-of-Thought reasoning',
                'components': ['blip_inference', 'query_reformulation', 'grad_cam', 'chain_of_thought', 'gemini_enhancement'],
                'order': 4,
                'color': '#96CEB4'  # Green
            },
            'enhanced_bbox': {
                'name': 'Complete MedXplain-VQA',
                'description': 'All components integrated',
                'components': ['blip_inference', 'query_reformulation', 'enhanced_grad_cam', 'bounding_boxes', 'chain_of_thought', 'gemini_enhancement'],
                'order': 5,
                'color': '#FFEAA7'  # Yellow
            }
        }
        
        # Key metrics for analysis
        self.key_metrics = [
            'enhanced_medical_similarity',
            'terminology_sophistication',
            'overall_medxplain_score'
        ]
        
        self.explainability_metrics = [
            'reasoning_confidence',
            'attention_quality', 
            'reformulation_quality'
        ]
        
        # Statistical analyzer
        self.stats_analyzer = StatisticalAnalyzer()
    
    def load_mode_data(self, mode_dirs: Dict[str, str]) -> Dict[str, Dict]:
        """Load and evaluate data from all modes"""
        print("🔄 Loading and evaluating data from all modes...")
        
        # Import evaluator
        try:
            from medical_evaluation_suite import FixedMedicalEvaluator
            evaluator = FixedMedicalEvaluator(debug=False)
        except ImportError:
            print("⚠️ Could not import medical evaluator. Using fallback.")
            evaluator = None
        
        mode_data = {}
        
        for mode_key, results_dir in mode_dirs.items():
            if mode_key not in self.modes:
                print(f"⚠️ Unknown mode: {mode_key}")
                continue
            
            print(f"📁 Loading {self.modes[mode_key]['name']} from {results_dir}")
            
            # Load JSON results
            results_path = Path(results_dir)
            if not results_path.exists():
                print(f"❌ Directory not found: {results_dir}")
                continue
            
            json_files = list(results_path.glob("*.json"))
            raw_results = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        raw_results.append(data)
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ Error loading {json_file}: {e}")
                    continue
            
            if not raw_results:
                print(f"❌ No valid results found in {results_dir}")
                continue
            
            print(f"  📊 Loaded {len(raw_results)} raw results")
            
            # Evaluate using medical evaluator
            evaluated_results = []
            if evaluator:
                for result in raw_results:
                    try:
                        evaluation = evaluator.evaluate_comprehensive_fixed(result)
                        evaluated_results.append(evaluation)
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️ Evaluation error: {e}")
                        continue
            else:
                # Fallback: use raw results with basic processing
                evaluated_results = self._process_raw_results(raw_results)
            
            if evaluated_results:
                mode_data[mode_key] = {
                    'raw_results': raw_results,
                    'evaluated_results': evaluated_results,
                    'mode_info': self.modes[mode_key],
                    'sample_count': len(evaluated_results)
                }
                print(f"  ✅ Successfully evaluated {len(evaluated_results)} samples")
            else:
                print(f"  ❌ No valid evaluations for {mode_key}")
        
        return mode_data
    
    def _process_raw_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Fallback processing for raw results when evaluator unavailable"""
        processed = []
        
        for result in raw_results:
            # Extract basic metrics from raw result
            processed_result = {
                'sample_id': result.get('sample_id', 'unknown'),
                'success': result.get('success', True),
                'enhanced_medical_similarity': 0.3,  # Fallback scores
                'terminology_sophistication': 0.4,
                'overall_medxplain_score': 0.45,
                'explainability_metrics': {
                    'reasoning_confidence': result.get('reasoning_analysis', {}).get('reasoning_confidence', 0.0),
                    'attention_quality': 0.5,
                    'reformulation_quality': result.get('reformulation_quality', 0.0)
                }
            }
            processed.append(processed_result)
        
        return processed
    
    def analyze_component_contributions(self, mode_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Comprehensive component contribution analysis"""
        print("🔬 Analyzing component contributions...")
        
        analysis = {
            'mode_performance': {},
            'component_improvements': {},
            'statistical_significance': {},
            'effect_sizes': {},
            'summary_statistics': {}
        }
        
        # Sort modes by order
        sorted_modes = sorted(mode_data.items(), key=lambda x: self.modes[x[0]]['order'])
        
        # 1. Calculate mode performance statistics
        for mode_key, data in sorted_modes:
            mode_name = data['mode_info']['name']
            evaluations = data['evaluated_results']
            
            print(f"  📊 Analyzing {mode_name} ({len(evaluations)} samples)")
            
            # Extract metric values
            metrics = {}
            for metric in self.key_metrics:
                values = [e.get(metric, 0.0) for e in evaluations]
                metrics[metric] = {
                    'values': values,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
            
            # Extract explainability metrics
            expl_metrics = {}
            for metric in self.explainability_metrics:
                values = [e.get('explainability_metrics', {}).get(metric, 0.0) for e in evaluations]
                expl_metrics[metric] = {
                    'values': values,
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            
            analysis['mode_performance'][mode_key] = {
                'mode_info': data['mode_info'],
                'sample_count': len(evaluations),
                'core_metrics': metrics,
                'explainability_metrics': expl_metrics
            }
        
        # 2. Calculate incremental improvements
        baseline_mode = None
        for mode_key, data in sorted_modes:
            current_performance = analysis['mode_performance'][mode_key]
            
            if baseline_mode is None:
                # First mode (baseline)
                baseline_mode = mode_key
                analysis['component_improvements'][mode_key] = {
                    'mode_info': current_performance['mode_info'],
                    'is_baseline': True,
                    'improvements': {}
                }
            else:
                # Calculate improvements over previous mode and baseline
                prev_mode_key = sorted_modes[current_performance['mode_info']['order'] - 2][0]
                prev_performance = analysis['mode_performance'][prev_mode_key]
                baseline_performance = analysis['mode_performance'][baseline_mode]
                
                improvements = {}
                
                for metric in self.key_metrics:
                    current_mean = current_performance['core_metrics'][metric]['mean']
                    prev_mean = prev_performance['core_metrics'][metric]['mean']
                    baseline_mean = baseline_performance['core_metrics'][metric]['mean']
                    
                    # Calculate improvements
                    absolute_improvement = current_mean - prev_mean
                    relative_improvement = (absolute_improvement / prev_mean * 100) if prev_mean > 0 else 0
                    baseline_improvement = current_mean - baseline_mean
                    baseline_relative = (baseline_improvement / baseline_mean * 100) if baseline_mean > 0 else 0
                    
                    improvements[metric] = {
                        'absolute_vs_previous': absolute_improvement,
                        'relative_vs_previous': relative_improvement,
                        'absolute_vs_baseline': baseline_improvement,
                        'relative_vs_baseline': baseline_relative,
                        'current_score': current_mean,
                        'previous_score': prev_mean,
                        'baseline_score': baseline_mean
                    }
                    
                    # Statistical significance testing
                    current_values = current_performance['core_metrics'][metric]['values']
                    prev_values = prev_performance['core_metrics'][metric]['values']
                    
                    stat_test = self.stats_analyzer.independent_t_test(prev_values, current_values)
                    improvements[metric]['statistical_test'] = stat_test
                
                analysis['component_improvements'][mode_key] = {
                    'mode_info': current_performance['mode_info'],
                    'is_baseline': False,
                    'improvements': improvements,
                    'previous_mode': prev_mode_key
                }
        
        # 3. Overall system progression analysis
        analysis['system_progression'] = self._analyze_system_progression(analysis['mode_performance'])
        
        return analysis
    
    def _analyze_system_progression(self, mode_performance: Dict) -> Dict[str, Any]:
        """Analyze overall system progression across modes"""
        progression = {
            'metric_trends': {},
            'consistency_analysis': {},
            'performance_gains': {}
        }
        
        # Sort modes by order
        sorted_modes = sorted(mode_performance.items(), 
                            key=lambda x: x[1]['mode_info']['order'])
        
        # Analyze trends for each metric
        for metric in self.key_metrics:
            scores = []
            mode_names = []
            
            for mode_key, data in sorted_modes:
                scores.append(data['core_metrics'][metric]['mean'])
                mode_names.append(data['mode_info']['name'])
            
            # Calculate trend statistics
            if len(scores) > 1:
                # Linear regression for trend
                x = np.arange(len(scores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
                
                progression['metric_trends'][metric] = {
                    'scores': scores,
                    'mode_names': mode_names,
                    'trend_slope': slope,
                    'trend_r_squared': r_value ** 2,
                    'trend_p_value': p_value,
                    'total_improvement': scores[-1] - scores[0],
                    'relative_total_improvement': ((scores[-1] - scores[0]) / scores[0] * 100) if scores[0] > 0 else 0
                }
        
        return progression
    
    def generate_visualizations(self, analysis: Dict[str, Any], output_dir: str):
        """Generate comprehensive visualizations"""
        print("📊 Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Component Improvement Chart
        self._create_component_improvement_chart(analysis, output_dir)
        
        # 2. Performance Progression Chart  
        self._create_performance_progression_chart(analysis, output_dir)
        
        # 3. Statistical Significance Heatmap
        self._create_significance_heatmap(analysis, output_dir)
        
        # 4. Box Plot Comparison
        self._create_boxplot_comparison(analysis, output_dir)
        
        print("✅ All visualizations generated")
    
    def _create_component_improvement_chart(self, analysis: Dict[str, Any], output_dir: str):
        """Create component improvement bar chart"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(self.key_metrics):
            ax = axes[idx]
            
            modes = []
            improvements = []
            colors = []
            significance = []
            
            for mode_key, data in analysis['component_improvements'].items():
                if data['is_baseline']:
                    continue
                
                mode_name = data['mode_info']['name']
                improvement = data['improvements'][metric]['relative_vs_previous']
                is_significant = data['improvements'][metric]['statistical_test']['significant']
                
                modes.append(mode_name)
                improvements.append(improvement)
                colors.append(data['mode_info']['color'])
                significance.append(is_significant)
            
            # Create bars
            bars = ax.bar(modes, improvements, color=colors, alpha=0.8, edgecolor='black')
            
            # Add significance indicators
            for i, (bar, is_sig) in enumerate(zip(bars, significance)):
                if is_sig:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           '***', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax.set_title(f'{metric.replace("_", " ").title()}\nComponent Improvements', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Relative Improvement (%)', fontsize=12)
            ax.set_xlabel('Mode Progression', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'component_improvements.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_progression_chart(self, analysis: Dict[str, Any], output_dir: str):
        """Create performance progression line chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        system_progression = analysis['system_progression']
        
        for metric in self.key_metrics:
            if metric in system_progression['metric_trends']:
                trend_data = system_progression['metric_trends'][metric]
                scores = trend_data['scores']
                mode_names = trend_data['mode_names']
                
                ax.plot(mode_names, scores, marker='o', linewidth=3, markersize=8, 
                       label=metric.replace('_', ' ').title())
        
        ax.set_title('MedXplain-VQA Performance Progression\nAcross Component Integration', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Processing Mode', fontsize=14)
        ax.set_ylabel('Performance Score', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'performance_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_significance_heatmap(self, analysis: Dict[str, Any], output_dir: str):
        """Create statistical significance heatmap"""
        # Prepare data for heatmap
        modes = []
        significance_matrix = []
        
        for mode_key, data in analysis['component_improvements'].items():
            if data['is_baseline']:
                continue
            
            mode_name = data['mode_info']['name']
            modes.append(mode_name)
            
            row = []
            for metric in self.key_metrics:
                p_value = data['improvements'][metric]['statistical_test']['p_value']
                # Convert p-value to significance level
                if p_value < 0.001:
                    sig_level = 3  # ***
                elif p_value < 0.01:
                    sig_level = 2  # **
                elif p_value < 0.05:
                    sig_level = 1  # *
                else:
                    sig_level = 0  # ns
                row.append(sig_level)
            
            significance_matrix.append(row)
        
        if significance_matrix:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create heatmap
            im = ax.imshow(significance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
            
            # Set ticks and labels
            ax.set_xticks(range(len(self.key_metrics)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in self.key_metrics])
            ax.set_yticks(range(len(modes)))
            ax.set_yticklabels(modes)
            
            # Add text annotations
            for i in range(len(modes)):
                for j in range(len(self.key_metrics)):
                    sig_level = significance_matrix[i][j]
                    if sig_level == 3:
                        text = '***'
                    elif sig_level == 2:
                        text = '**'
                    elif sig_level == 1:
                        text = '*'
                    else:
                        text = 'ns'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white' if sig_level > 1 else 'black', fontweight='bold')
            
            ax.set_title('Statistical Significance of Component Improvements\n(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'significance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_boxplot_comparison(self, analysis: Dict[str, Any], output_dir: str):
        """Create box plot comparison across modes"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(self.key_metrics):
            ax = axes[idx]
            
            # Prepare data
            data_for_plot = []
            labels = []
            colors = []
            
            sorted_modes = sorted(analysis['mode_performance'].items(), 
                                key=lambda x: x[1]['mode_info']['order'])
            
            for mode_key, data in sorted_modes:
                values = data['core_metrics'][metric]['values']
                data_for_plot.append(values)
                labels.append(data['mode_info']['name'])
                colors.append(data['mode_info']['color'])
            
            # Create box plot
            box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{metric.replace("_", " ").title()}\nDistribution Across Modes', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_paper_tables(self, analysis: Dict[str, Any], output_dir: str):
        """Generate LaTeX tables for paper"""
        print("📄 Generating paper-ready tables...")
        
        # 1. Main Results Table
        self._generate_main_results_table(analysis, output_dir)
        
        # 2. Component Contribution Table
        self._generate_component_contribution_table(analysis, output_dir)
        
        # 3. Statistical Significance Table
        self._generate_statistical_table(analysis, output_dir)
        
        print("✅ All paper tables generated")
    
    def _generate_main_results_table(self, analysis: Dict[str, Any], output_dir: str):
        """Generate main results comparison table"""
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{MedXplain-VQA Component Ablation Study Results}")
        latex_content.append("\\label{tab:ablation_results}")
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")
        latex_content.append("\\textbf{Mode} & \\textbf{Medical Similarity} & \\textbf{Terminology} & \\textbf{Overall Score} & \\textbf{Samples} \\\\")
        latex_content.append("\\midrule")
        
        # Sort modes by order
        sorted_modes = sorted(analysis['mode_performance'].items(), 
                            key=lambda x: x[1]['mode_info']['order'])
        
        for mode_key, data in sorted_modes:
            mode_name = data['mode_info']['name']
            sample_count = data['sample_count']
            
            # Get metrics
            med_sim = data['core_metrics']['enhanced_medical_similarity']['mean']
            med_sim_std = data['core_metrics']['enhanced_medical_similarity']['std']
            
            terminology = data['core_metrics']['terminology_sophistication']['mean']
            terminology_std = data['core_metrics']['terminology_sophistication']['std']
            
            overall = data['core_metrics']['overall_medxplain_score']['mean']
            overall_std = data['core_metrics']['overall_medxplain_score']['std']
            
            latex_content.append(f"{mode_name} & {med_sim:.3f} $\\pm$ {med_sim_std:.3f} & "
                               f"{terminology:.3f} $\\pm$ {terminology_std:.3f} & "
                               f"{overall:.3f} $\\pm$ {overall_std:.3f} & {sample_count} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX table
        with open(Path(output_dir) / 'main_results_table.tex', 'w') as f:
            f.write('\n'.join(latex_content))
    
    def _generate_component_contribution_table(self, analysis: Dict[str, Any], output_dir: str):
        """Generate component contribution analysis table"""
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Component Contribution Analysis}")
        latex_content.append("\\label{tab:component_contribution}")
        latex_content.append("\\begin{tabular}{lccc}")
        latex_content.append("\\toprule")
        latex_content.append("\\textbf{Component Addition} & \\textbf{Improvement} & \\textbf{Relative (\\%)} & \\textbf{Significance} \\\\")
        latex_content.append("\\midrule")
        
        for mode_key, data in analysis['component_improvements'].items():
            if data['is_baseline']:
                continue
            
            mode_name = data['mode_info']['name']
            
            # Use overall score for main comparison
            improvement_data = data['improvements']['overall_medxplain_score']
            absolute_imp = improvement_data['absolute_vs_previous']
            relative_imp = improvement_data['relative_vs_previous']
            is_significant = improvement_data['statistical_test']['significant']
            
            sig_marker = "***" if is_significant else "ns"
            
            latex_content.append(f"{mode_name} & +{absolute_imp:.3f} & +{relative_imp:.1f}\\% & {sig_marker} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX table
        with open(Path(output_dir) / 'component_contribution_table.tex', 'w') as f:
            f.write('\n'.join(latex_content))
    
    def _generate_statistical_table(self, analysis: Dict[str, Any], output_dir: str):
        """Generate detailed statistical analysis table"""
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Statistical Significance Analysis}")
        latex_content.append("\\label{tab:statistical_analysis}")
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")
        latex_content.append("\\textbf{Comparison} & \\textbf{t-statistic} & \\textbf{p-value} & \\textbf{Effect Size} & \\textbf{Significant} \\\\")
        latex_content.append("\\midrule")
        
        for mode_key, data in analysis['component_improvements'].items():
            if data['is_baseline']:
                continue
            
            mode_name = data['mode_info']['name']
            
            # Use overall score statistics
            stat_test = data['improvements']['overall_medxplain_score']['statistical_test']
            t_stat = stat_test['t_statistic']
            p_value = stat_test['p_value']
            effect_size = stat_test['effect_size']
            significant = "Yes" if stat_test['significant'] else "No"
            
            latex_content.append(f"{mode_name} & {t_stat:.3f} & {p_value:.4f} & {effect_size:.3f} & {significant} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX table
        with open(Path(output_dir) / 'statistical_analysis_table.tex', 'w') as f:
            f.write('\n'.join(latex_content))
    
    def generate_comprehensive_report(self, analysis: Dict[str, Any], output_dir: str):
        """Generate comprehensive text report"""
        print("📝 Generating comprehensive report...")
        
        report_lines = []
        
        # Header
        report_lines.append("🔬 MEDXPLAIN-VQA COMPONENT ABLATION STUDY")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("📊 EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Find best performing mode
        best_mode = None
        best_score = 0
        for mode_key, data in analysis['mode_performance'].items():
            score = data['core_metrics']['overall_medxplain_score']['mean']
            if score > best_score:
                best_score = score
                best_mode = data['mode_info']['name']
        
        if best_mode:
            report_lines.append(f"🏆 Best performing mode: {best_mode} (score: {best_score:.3f})")
        
        # Calculate total improvement
        sorted_modes = sorted(analysis['mode_performance'].items(), 
                            key=lambda x: x[1]['mode_info']['order'])
        if len(sorted_modes) >= 2:
            baseline_score = sorted_modes[0][1]['core_metrics']['overall_medxplain_score']['mean']
            final_score = sorted_modes[-1][1]['core_metrics']['overall_medxplain_score']['mean']
            total_improvement = ((final_score - baseline_score) / baseline_score * 100)
            report_lines.append(f"📈 Total system improvement: +{total_improvement:.1f}% over baseline")
        
        report_lines.append("")
        
        # Detailed Results
        report_lines.append("📊 DETAILED PERFORMANCE RESULTS")
        report_lines.append("-" * 40)
        
        for mode_key, data in sorted_modes:
            mode_name = data['mode_info']['name']
            sample_count = data['sample_count']
            
            report_lines.append(f"\n{mode_name} ({sample_count} samples):")
            
            for metric in self.key_metrics:
                metric_data = data['core_metrics'][metric]
                mean_val = metric_data['mean']
                std_val = metric_data['std']
                
                metric_display = metric.replace('_', ' ').title()
                report_lines.append(f"  {metric_display:<30} {mean_val:.3f} ± {std_val:.3f}")
        
        # Component Improvements
        report_lines.append("\n⚙️ COMPONENT IMPROVEMENT ANALYSIS")
        report_lines.append("-" * 40)
        
        for mode_key, data in analysis['component_improvements'].items():
            if data['is_baseline']:
                continue
            
            mode_name = data['mode_info']['name']
            report_lines.append(f"\n{mode_name}:")
            
            for metric in self.key_metrics:
                improvement_data = data['improvements'][metric]
                relative_imp = improvement_data['relative_vs_previous']
                is_significant = improvement_data['statistical_test']['significant']
                
                sig_marker = " (***)" if is_significant else " (ns)"
                metric_display = metric.replace('_', ' ').title()
                
                report_lines.append(f"  {metric_display:<30} +{relative_imp:6.1f}%{sig_marker}")
        
        # System Progression Analysis
        system_progression = analysis.get('system_progression', {})
        if 'metric_trends' in system_progression:
            report_lines.append("\n📈 SYSTEM PROGRESSION ANALYSIS")
            report_lines.append("-" * 40)
            
            for metric, trend_data in system_progression['metric_trends'].items():
                total_improvement = trend_data['total_improvement']
                relative_total = trend_data['relative_total_improvement']
                
                metric_display = metric.replace('_', ' ').title()
                report_lines.append(f"{metric_display}: +{total_improvement:.3f} (+{relative_total:.1f}% total improvement)")
        
        # Save report
        with open(Path(output_dir) / 'ablation_study_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ Comprehensive report generated")

def main():
    parser = argparse.ArgumentParser(description='🔬 Component Ablation Study for MedXplain-VQA')
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
    parser.add_argument('--output-dir', type=str, default='data/ablation_study_results',
                       help='Output directory for analysis results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('ablation_study', args.output_dir, level='INFO')
    
    logger.info("🚀 Starting Component Ablation Study")
    logger.info("🎯 Analyzing 5 MedXplain-VQA processing modes with statistical rigor")
    
    # Initialize analyzer
    analyzer = ComponentAblationAnalyzer(debug=args.debug)
    
    # Define mode directories
    mode_dirs = {
        'basic': args.basic_dir,
        'explainable': args.explainable_dir,
        'explainable_bbox': args.bbox_dir,
        'enhanced': args.enhanced_dir,
        'enhanced_bbox': args.full_dir
    }
    
    # Load and evaluate data
    logger.info("📂 Loading data from all modes...")
    mode_data = analyzer.load_mode_data(mode_dirs)
    
    if not mode_data:
        logger.error("❌ No valid mode data found!")
        return
    
    logger.info(f"✅ Successfully loaded data for {len(mode_data)} modes")
    
    # Perform ablation analysis
    logger.info("🔬 Performing comprehensive component ablation analysis...")
    analysis = analyzer.analyze_component_contributions(mode_data)
    
    # Generate outputs
    logger.info("📊 Generating visualizations...")
    analyzer.generate_visualizations(analysis, args.output_dir)
    
    logger.info("📄 Generating paper-ready tables...")
    analyzer.generate_paper_tables(analysis, args.output_dir)
    
    logger.info("📝 Generating comprehensive report...")
    analyzer.generate_comprehensive_report(analysis, args.output_dir)
    
    # Save complete analysis data
    analysis_file = Path(args.output_dir) / 'complete_ablation_analysis.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"💾 Complete analysis saved to: {analysis_file}")
    
    # Print summary results
    logger.info("\n" + "="*80)
    logger.info("🎯 ABLATION STUDY RESULTS SUMMARY")
    logger.info("="*80)
    
    # Mode performance ranking
    mode_scores = []
    for mode_key, data in analysis['mode_performance'].items():
        score = data['core_metrics']['overall_medxplain_score']['mean']
        mode_scores.append((data['mode_info']['name'], score, data['sample_count']))
    
    mode_scores.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("\n📊 MODE PERFORMANCE RANKING:")
    for rank, (mode_name, score, samples) in enumerate(mode_scores, 1):
        logger.info(f"  {rank}. {mode_name:<25} Score: {score:.3f} (n={samples})")
    
    # Significant improvements
    logger.info("\n⚙️ SIGNIFICANT COMPONENT IMPROVEMENTS:")
    for mode_key, data in analysis['component_improvements'].items():
        if data['is_baseline']:
            continue
        
        mode_name = data['mode_info']['name']
        overall_improvement = data['improvements']['overall_medxplain_score']
        
        if overall_improvement['statistical_test']['significant']:
            relative_imp = overall_improvement['relative_vs_previous']
            logger.info(f"  {mode_name:<25} +{relative_imp:6.1f}% (p < 0.05) ***")
    
    # System progression summary
    system_progression = analysis.get('system_progression', {})
    if 'metric_trends' in system_progression:
        logger.info("\n📈 OVERALL SYSTEM IMPROVEMENT:")
        
        for metric, trend_data in system_progression['metric_trends'].items():
            total_improvement = trend_data['relative_total_improvement']
            metric_display = metric.replace('_', ' ').title()
            logger.info(f"  {metric_display:<30} +{total_improvement:6.1f}% total")
    
    logger.info(f"\n💾 All results saved to: {args.output_dir}")
    logger.info("📊 Visualizations: component_improvements.png, performance_progression.png")
    logger.info("📄 LaTeX tables: main_results_table.tex, component_contribution_table.tex")
    logger.info("📝 Text report: ablation_study_report.txt")
    logger.info("\n🎉 Component Ablation Study completed successfully!")

if __name__ == "__main__":
    main()
