#!/usr/bin/env python
"""
Paper Visualization Suite
=========================

Generate all figures and tables for research paper.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_paper_visualizations():
    """Create all paper visualizations"""
    
    # Load evaluation results
    with open('data/final_medical_evaluation/final_evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    output_dir = Path('data/paper_figures')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Performance Comparison Chart
    create_performance_comparison(results, output_dir)
    
    # 2. Component Contribution Analysis  
    create_component_analysis(results, output_dir)
    
    # 3. Explainability Features Radar Chart
    create_explainability_radar(results, output_dir)
    
    # 4. Statistical Significance Table
    create_statistical_table(results, output_dir)
    
    print("✅ All paper visualizations created!")

def create_performance_comparison(results, output_dir):
    """Create performance comparison bar chart"""
    modes = ['basic', 'explainable', 'explainable_bbox', 'enhanced', 'enhanced_bbox']
    
    # Extract metrics
    explanation_quality = []
    medical_terms = []
    coherence = []
    
    for mode in modes:
        metrics = results[mode]['vqa_metrics']
        explanation_quality.append(metrics['explanation_length']['mean'])
        medical_terms.append(metrics['medical_terminology']['mean'])
        coherence.append(metrics['explanation_coherence']['mean'])
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(modes))
    width = 0.6
    
    # Explanation Quality
    bars1 = ax1.bar(x, explanation_quality, width, color='skyblue', alpha=0.8)
    ax1.set_title('Explanation Length Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in modes], rotation=0)
    ax1.set_ylim(0, 1.1)
    
    # Medical Terminology
    bars2 = ax2.bar(x, medical_terms, width, color='lightgreen', alpha=0.8)
    ax2.set_title('Medical Terminology Usage', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in modes], rotation=0)
    ax2.set_ylim(0, 0.5)
    
    # Coherence
    bars3 = ax3.bar(x, coherence, width, color='salmon', alpha=0.8)
    ax3.set_title('Explanation Coherence', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('_', '\n') for m in modes], rotation=0)
    ax3.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_component_analysis(results, output_dir):
    """Create component contribution analysis"""
    # Component progression
    components = ['Basic', '+ Query\nReform', '+ Attention', '+ Bounding\nBoxes', '+ Chain-of\nThought']
    modes = ['basic', 'explainable', 'explainable', 'explainable_bbox', 'enhanced_bbox']
    
    # Calculate composite scores
    composite_scores = []
    for mode in modes:
        if mode in results:
            metrics = results[mode]['vqa_metrics']
            # Weighted composite score
            score = (metrics['explanation_length']['mean'] * 0.2 +
                    metrics['medical_terminology']['mean'] * 0.3 +
                    metrics['clinical_structure']['mean'] * 0.2 +
                    metrics['explanation_coherence']['mean'] * 0.3)
            composite_scores.append(score)
        else:
            composite_scores.append(0)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Bar chart with progression
    bars = plt.bar(components, composite_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'], alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, composite_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Component Contribution Analysis\n(Weighted Composite Score)', fontsize=16, fontweight='bold')
    plt.ylabel('Composite Score', fontsize=12)
    plt.xlabel('MedXplain-VQA Components', fontsize=12)
    plt.ylim(0, max(composite_scores) * 1.2)
    
    # Add improvement arrows
    for i in range(1, len(composite_scores)):
        improvement = ((composite_scores[i] - composite_scores[i-1]) / composite_scores[i-1]) * 100
        plt.annotate(f'+{improvement:.1f}%', xy=(i, composite_scores[i] + 0.02), 
                    ha='center', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_explainability_radar(results, output_dir):
    """Create explainability features radar chart"""
    modes = ['basic', 'explainable_bbox', 'enhanced_bbox']
    mode_labels = ['Basic', 'Explainable + BBox', 'Enhanced + BBox']
    
    # Features to compare
    features = ['Attention\nCoverage', 'Attention\nQuality', 'Reasoning\nConfidence', 
               'Medical\nTerminology', 'Clinical\nStructure']
    
    # Extract data
    data = []
    for mode in modes:
        if mode in results:
            vqa_metrics = results[mode]['vqa_metrics']
            exp_metrics = results[mode]['explainability_metrics']
            
            row = [
                exp_metrics.get('attention_coverage', {}).get('mean', 0),
                exp_metrics.get('attention_quality', {}).get('mean', 0),
                exp_metrics.get('reasoning_confidence', {}).get('mean', 0),
                vqa_metrics.get('medical_terminology', {}).get('mean', 0),
                vqa_metrics.get('clinical_structure', {}).get('mean', 0)
            ]
            data.append(row)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFEAA7']
    
    for i, (mode_data, label, color) in enumerate(zip(data, mode_labels, colors)):
        mode_data = mode_data + [mode_data[0]]  # Complete the circle
        ax.plot(angles, mode_data, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, mode_data, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('MedXplain-VQA Explainability Features\n(Radar Chart Comparison)', 
                size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'explainability_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_table(results, output_dir):
    """Create statistical significance table"""
    # Create LaTeX table for paper
    table_data = []
    
    for mode_name, evaluation in results.items():
        if 'vqa_metrics' in evaluation:
            vqa_metrics = evaluation['vqa_metrics']
            exp_metrics = evaluation['explainability_metrics']
            
            row = {
                'Mode': mode_name.replace('_', '\\_'),
                'Samples': evaluation['sample_count'],
                'Medical Terms': f"{vqa_metrics.get('medical_terminology', {}).get('mean', 0):.3f} ± {vqa_metrics.get('medical_terminology', {}).get('std', 0):.3f}",
                'Coherence': f"{vqa_metrics.get('explanation_coherence', {}).get('mean', 0):.3f} ± {vqa_metrics.get('explanation_coherence', {}).get('std', 0):.3f}",
                'Attention Quality': f"{exp_metrics.get('attention_quality', {}).get('mean', 0):.3f}",
                'Reasoning Conf': f"{exp_metrics.get('reasoning_confidence', {}).get('mean', 0):.3f}"
            }
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV for easy conversion to LaTeX
    df.to_csv(output_dir / 'statistical_table.csv', index=False)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    with open(output_dir / 'statistical_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("📊 Statistical table saved as CSV and LaTeX")

if __name__ == "__main__":
    create_paper_visualizations()
