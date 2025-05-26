#!/usr/bin/env python
"""
Ablation Study Analysis
"""

import json
import pandas as pd
from pathlib import Path

def analyze_component_contributions():
    with open('data/final_medical_evaluation/final_evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Component progression analysis
    progression = {
        'Basic': results['basic']['vqa_metrics']['medical_terminology']['mean'],
        '+ Query Reform': results['explainable']['vqa_metrics']['medical_terminology']['mean'],  
        '+ Attention': results['explainable']['vqa_metrics']['medical_terminology']['mean'],
        '+ BBox': results['explainable_bbox']['vqa_metrics']['medical_terminology']['mean'],
        '+ CoT': results['enhanced_bbox']['vqa_metrics']['medical_terminology']['mean']
    }
    
    # Calculate improvements
    baseline = progression['Basic']
    improvements = {}
    for component, score in progression.items():
        improvement = ((score - baseline) / baseline) * 100
        improvements[component] = improvement
        print(f"{component}: {score:.3f} (+{improvement:.1f}%)")
    
    return improvements

if __name__ == "__main__":
    analyze_component_contributions()
