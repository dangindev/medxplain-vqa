#!/usr/bin/env python
"""
Final Ablation Analysis with Composite Scoring
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def create_comprehensive_ablation():
    """Create comprehensive ablation study with composite scoring"""
    
    with open('data/final_fixed_medical_evaluation/final_fixed_evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Define component progression
    components = {
        'Basic (BLIP + Gemini)': 'basic',
        '+ Query Reformulation': 'explainable', 
        '+ Attention Mechanisms': 'explainable',  # Same as explainable
        '+ Bounding Box Detection': 'explainable_bbox',
        '+ Chain-of-Thought': 'enhanced',
        '+ Complete System (BBox + CoT)': 'enhanced_bbox'
    }
    
    print("🔍 COMPREHENSIVE COMPONENT ANALYSIS")
    print("=" * 80)
    
    # Calculate composite scores
    composite_scores = []
    detailed_breakdown = []
    
    for component_name, mode_key in components.items():
        if mode_key in results:
            vqa_metrics = results[mode_key]['vqa_metrics']
            exp_metrics = results[mode_key]['explainability_metrics']
            
            # Individual metrics
            med_terminology = vqa_metrics.get('medical_terminology', {}).get('mean', 0)
            clinical_structure = vqa_metrics.get('clinical_structure', {}).get('mean', 0)
            coherence = vqa_metrics.get('explanation_coherence', {}).get('mean', 0)
            attention_quality = exp_metrics.get('attention_quality', {}).get('mean', 0)
            reasoning_confidence = exp_metrics.get('reasoning_confidence', {}).get('mean', 0)
            
            # COMPOSITE SCORE CALCULATION
            # Weight different aspects appropriately
            composite = (
                med_terminology * 0.25 +      # Medical terminology (25%)
                clinical_structure * 0.20 +   # Clinical structure (20%)
                coherence * 0.25 +            # Explanation coherence (25%)
                attention_quality * 0.15 +    # Attention quality (15%)
                reasoning_confidence * 0.15   # Reasoning confidence (15%)
            )
            
            composite_scores.append(composite)
            
            detailed_breakdown.append({
                'Component': component_name,
                'Mode': mode_key,
                'Medical Terms': f"{med_terminology:.3f}",
                'Clinical Struct': f"{clinical_structure:.3f}",
                'Coherence': f"{coherence:.3f}",
                'Attention': f"{attention_quality:.3f}",
                'Reasoning': f"{reasoning_confidence:.3f}",
                'Composite': f"{composite:.3f}"
            })
            
            print(f"\n📊 {component_name}")
            print(f"   Medical Terminology: {med_terminology:.3f}")
            print(f"   Clinical Structure:  {clinical_structure:.3f}")
            print(f"   Explanation Coherence: {coherence:.3f}")
            print(f"   Attention Quality:   {attention_quality:.3f}")
            print(f"   Reasoning Confidence: {reasoning_confidence:.3f}")
            print(f"   🎯 COMPOSITE SCORE:   {composite:.3f}")
    
    # Calculate improvements  
    print(f"\n🚀 COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 50)
    
    baseline_score = composite_scores[0]  # Basic mode
    
    for i, (component_name, score) in enumerate(zip(components.keys(), composite_scores)):
        if i == 0:
            improvement = 0.0
            print(f"{component_name:30} | {score:.3f} | Baseline")
        else:
            improvement = ((score - baseline_score) / baseline_score) * 100
            delta = score - composite_scores[i-1]
            print(f"{component_name:30} | {score:.3f} | +{improvement:5.1f}% | Δ{delta:+.3f}")
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame(detailed_breakdown)
    print(f"\n📋 DETAILED BREAKDOWN TABLE")
    print("=" * 100)
    print(df.to_string(index=False))
    
    # Key insights
    best_component = max(zip(components.keys(), composite_scores), key=lambda x: x[1])
    print(f"\n💡 KEY INSIGHTS")
    print("=" * 30)
    print(f"🏆 Best Overall Performance: {best_component[0]} (Score: {best_component[1]:.3f})")
    print(f"📈 Total Improvement: +{((composite_scores[-1] - composite_scores[0]) / composite_scores[0] * 100):.1f}%")
    
    # Individual component contributions
    print(f"\n🔍 INDIVIDUAL COMPONENT IMPACTS:")
    component_impacts = []
    for i in range(1, len(composite_scores)):
        delta = composite_scores[i] - composite_scores[i-1]
        component_impacts.append((list(components.keys())[i], delta))
    
    # Sort by impact
    component_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    for component, impact in component_impacts:
        impact_sign = "📈" if impact > 0 else "📉"
        print(f"   {impact_sign} {component}: {impact:+.3f}")
    
    return df, composite_scores

if __name__ == "__main__":
    df, scores = create_comprehensive_ablation()
    
    # Save results
    output_dir = Path('data/final_ablation_analysis')
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'comprehensive_ablation_results.csv', index=False)
    
    print(f"\n✅ Comprehensive ablation analysis saved to {output_dir}")
