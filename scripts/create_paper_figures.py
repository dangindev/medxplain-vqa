#!/usr/bin/env python
"""
Figure 1: MedXplain-VQA System Overview for Paper Introduction
============================================================

Creates comprehensive overview figure showing:
1. Problem motivation (traditional vs explainable VQA)
2. MedXplain-VQA system pipeline
3. Key innovations and outputs

For Paper Section 1 (Introduction)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_introduction_overview_figure(output_path="paper_figures/figure1_introduction_overview.png"):
    """
    Create Figure 1 for paper introduction showing system overview
    """
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 14))
    
    # Define color scheme
    colors = {
        'traditional': '#FFE5E5',  # Light red
        'medxplain': '#E5F7FF',   # Light blue  
        'innovation': '#E5FFE5',   # Light green
        'attention': '#FFF5E5',    # Light orange
        'text': '#2C3E50',         # Dark blue-gray
        'border': '#34495E',       # Darker gray
        'arrow': '#E74C3C',        # Red
        'success': '#27AE60'       # Green
    }
    
    # Create grid layout
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 0.3, 1.5, 1], width_ratios=[1, 0.2, 1, 0.2, 1, 1])
    
    # ============================================================================
    # TOP SECTION: PROBLEM MOTIVATION (Row 0)
    # ============================================================================
    
    # Traditional VQA (Left)
    ax_traditional = fig.add_subplot(gs[0, 0])
    ax_traditional.set_xlim(0, 10)
    ax_traditional.set_ylim(0, 10)
    
    # Traditional VQA box
    traditional_box = FancyBboxPatch(
        (0.5, 1), 9, 8,
        boxstyle="round,pad=0.3",
        facecolor=colors['traditional'],
        edgecolor=colors['border'],
        linewidth=2
    )
    ax_traditional.add_patch(traditional_box)
    
    # Traditional VQA content
    ax_traditional.text(5, 8, 'Traditional Medical VQA', 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       color=colors['text'])
    ax_traditional.text(5, 6.5, 'Input: Medical Image + Question', 
                       ha='center', va='center', fontsize=11, color=colors['text'])
    ax_traditional.text(5, 5.5, '↓', ha='center', va='center', fontsize=16, color=colors['arrow'])
    ax_traditional.text(5, 4.5, 'Black-box Model', 
                       ha='center', va='center', fontsize=11, color=colors['text'])
    ax_traditional.text(5, 3.5, '↓', ha='center', va='center', fontsize=16, color=colors['arrow'])
    ax_traditional.text(5, 2.5, 'Output: Answer Only', 
                       ha='center', va='center', fontsize=11, color=colors['text'])
    
    # Limitations
    ax_traditional.text(5, 1.2, '❌ No explanations\n❌ No attention\n❌ Limited trust', 
                       ha='center', va='top', fontsize=9, color='#C0392B')
    
    ax_traditional.set_title('(a) Traditional Approach', fontsize=12, fontweight='bold', pad=20)
    ax_traditional.axis('off')
    
    # Arrow between traditional and MedXplain
    ax_arrow = fig.add_subplot(gs[0, 1])
    ax_arrow.set_xlim(0, 2)
    ax_arrow.set_ylim(0, 10)
    
    arrow = patches.FancyArrowPatch(
        (0.5, 5), (1.5, 5),
        arrowstyle='->', mutation_scale=30,
        color=colors['arrow'], linewidth=3
    )
    ax_arrow.add_patch(arrow)
    ax_arrow.text(1, 6, 'Our\nApproach', ha='center', va='center', 
                 fontsize=11, fontweight='bold', color=colors['arrow'])
    ax_arrow.axis('off')
    
    # MedXplain-VQA (Right)
    ax_medxplain = fig.add_subplot(gs[0, 2])
    ax_medxplain.set_xlim(0, 10)
    ax_medxplain.set_ylim(0, 10)
    
    # MedXplain VQA box
    medxplain_box = FancyBboxPatch(
        (0.5, 1), 9, 8,
        boxstyle="round,pad=0.3",
        facecolor=colors['medxplain'],
        edgecolor=colors['border'],
        linewidth=2
    )
    ax_medxplain.add_patch(medxplain_box)
    
    # MedXplain VQA content
    ax_medxplain.text(5, 8, 'MedXplain-VQA (Ours)', 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     color=colors['text'])
    ax_medxplain.text(5, 6.5, 'Input: Medical Image + Question', 
                     ha='center', va='center', fontsize=11, color=colors['text'])
    ax_medxplain.text(5, 5.5, '↓', ha='center', va='center', fontsize=16, color=colors['success'])
    ax_medxplain.text(5, 4.5, 'Explainable AI Pipeline', 
                     ha='center', va='center', fontsize=11, color=colors['text'])
    ax_medxplain.text(5, 3.5, '↓', ha='center', va='center', fontsize=16, color=colors['success'])
    ax_medxplain.text(5, 2.5, 'Answer + Explanations + Attention', 
                     ha='center', va='center', fontsize=11, color=colors['text'])
    
    # Benefits
    ax_medxplain.text(5, 1.2, '✅ Visual explanations\n✅ Reasoning chains\n✅ Clinical trust', 
                     ha='center', va='top', fontsize=9, color=colors['success'])
    
    ax_medxplain.set_title('(b) MedXplain-VQA Approach', fontsize=12, fontweight='bold', pad=20)
    ax_medxplain.axis('off')
    
    # Key Innovations (Far right)
    ax_innovations = fig.add_subplot(gs[0, 4:6])
    ax_innovations.set_xlim(0, 10)
    ax_innovations.set_ylim(0, 10)
    
    # Innovations box
    innovations_box = FancyBboxPatch(
        (0.5, 1), 9, 8,
        boxstyle="round,pad=0.3",
        facecolor=colors['innovation'],
        edgecolor=colors['border'],
        linewidth=2
    )
    ax_innovations.add_patch(innovations_box)
    
    ax_innovations.text(5, 8.5, 'Key Innovations', 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       color=colors['text'])
    
    innovations_text = """
🔬 Fine-tuned BLIP2 on PathVQA
📝 Medical Query Reformulation  
🎯 Enhanced Grad-CAM + Bounding Boxes
🧠 Chain-of-Thought Medical Reasoning
🤖 LLM Integration for Unified Answers
"""
    ax_innovations.text(5, 4.5, innovations_text.strip(), 
                       ha='center', va='center', fontsize=11, 
                       color=colors['text'], linespacing=1.5)
    
    ax_innovations.set_title('(c) Technical Contributions', fontsize=12, fontweight='bold', pad=20)
    ax_innovations.axis('off')
    
    # ============================================================================
    # MIDDLE SECTION: SYSTEM PIPELINE (Row 2)
    # ============================================================================
    
    ax_pipeline = fig.add_subplot(gs[2, :])
    ax_pipeline.set_xlim(0, 30)
    ax_pipeline.set_ylim(0, 12)
    
    # Pipeline title
    ax_pipeline.text(15, 11, 'MedXplain-VQA System Pipeline', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color=colors['text'])
    
    # Pipeline components
    components = [
        {'name': 'Medical\nImage', 'x': 2, 'color': colors['traditional']},
        {'name': 'Question', 'x': 5, 'color': colors['traditional']},
        {'name': 'BLIP2\nFine-tuned', 'x': 9, 'color': colors['medxplain']},
        {'name': 'Query\nReformulation', 'x': 13, 'color': colors['innovation']},
        {'name': 'Enhanced\nGrad-CAM', 'x': 17, 'color': colors['attention']},
        {'name': 'Chain-of-\nThought', 'x': 21, 'color': colors['innovation']},
        {'name': 'Unified\nAnswer', 'x': 25, 'color': colors['success']},
        {'name': 'Explainable\nOutput', 'x': 28, 'color': colors['success']}
    ]
    
    # Draw pipeline components
    for i, comp in enumerate(components):
        # Component box
        comp_box = FancyBboxPatch(
            (comp['x']-1, 6), 2, 3,
            boxstyle="round,pad=0.2",
            facecolor=comp['color'],
            edgecolor=colors['border'],
            linewidth=1.5
        )
        ax_pipeline.add_patch(comp_box)
        
        # Component text
        ax_pipeline.text(comp['x'], 7.5, comp['name'], 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color=colors['text'])
        
        # Arrow to next component
        if i < len(components) - 1:
            next_comp = components[i + 1]
            arrow = patches.FancyArrowPatch(
                (comp['x'] + 1, 7.5), (next_comp['x'] - 1, 7.5),
                arrowstyle='->', mutation_scale=15,
                color=colors['arrow'], linewidth=2
            )
            ax_pipeline.add_patch(arrow)
    
    # Input examples (bottom)
    ax_pipeline.text(3.5, 4, 'Input Example:', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color=colors['text'])
    ax_pipeline.text(3.5, 3, 'Q: "What is present?"', ha='center', va='center', 
                    fontsize=10, color=colors['text'], style='italic')
    ax_pipeline.text(3.5, 2.3, '[Pathology Image]', ha='center', va='center', 
                    fontsize=10, color=colors['text'])
    
    # Output examples (bottom)
    ax_pipeline.text(26.5, 4, 'Output Example:', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color=colors['text'])
    ax_pipeline.text(26.5, 3.4, '• Answer: "Demodex folliculorum"', ha='center', va='center', 
                    fontsize=9, color=colors['text'])
    ax_pipeline.text(26.5, 3, '• Attention: [Bounding boxes]', ha='center', va='center', 
                    fontsize=9, color=colors['text'])
    ax_pipeline.text(26.5, 2.6, '• Reasoning: [6-step chain]', ha='center', va='center', 
                    fontsize=9, color=colors['text'])
    ax_pipeline.text(26.5, 2.2, '• Explanation: [Clinical analysis]', ha='center', va='center', 
                    fontsize=9, color=colors['text'])
    
    ax_pipeline.axis('off')
    
    # ============================================================================
    # BOTTOM SECTION: SAMPLE OUTPUT VISUALIZATION (Row 3)
    # ============================================================================
    
    ax_sample = fig.add_subplot(gs[3, :])
    ax_sample.set_xlim(0, 24)
    ax_sample.set_ylim(0, 8)
    
    # Sample output title
    ax_sample.text(12, 7.5, 'Sample Explainable Output Visualization', 
                  ha='center', va='center', fontsize=14, fontweight='bold',
                  color=colors['text'])
    
    # Create 4 sample panels (mimicking the 4-panel visualization)
    panels = [
        {'name': 'Original\nImage', 'x': 3, 'desc': 'Input medical\nhistopathology'},
        {'name': 'Bounding\nBoxes', 'x': 9, 'desc': '3-5 attention\nregions detected'},
        {'name': 'Attention\nHeatmap', 'x': 15, 'desc': 'Enhanced Grad-CAM\nvisualization'},
        {'name': 'Combined\nView', 'x': 21, 'desc': 'Overlay of attention\n+ bounding boxes'}
    ]
    
    for panel in panels:
        # Panel box
        panel_box = FancyBboxPatch(
            (panel['x']-1.5, 2), 3, 3,
            boxstyle="round,pad=0.1",
            facecolor='white',
            edgecolor=colors['border'],
            linewidth=1.5
        )
        ax_sample.add_patch(panel_box)
        
        # Panel title
        ax_sample.text(panel['x'], 5.5, panel['name'], 
                      ha='center', va='center', fontsize=10, fontweight='bold',
                      color=colors['text'])
        
        # Panel description
        ax_sample.text(panel['x'], 1.2, panel['desc'], 
                      ha='center', va='center', fontsize=8,
                      color=colors['text'])
        
        # Simulate image content
        if panel['name'] == 'Original\nImage':
            # Draw a simple medical image representation
            circle = Circle((panel['x'], 3.5), 0.8, color='#FFB6C1', alpha=0.7)
            ax_sample.add_patch(circle)
            ax_sample.text(panel['x'], 3.5, 'Medical\nTissue', ha='center', va='center', 
                          fontsize=8, color=colors['text'])
        
        elif panel['name'] == 'Bounding\nBoxes':
            # Draw bounding boxes
            circle = Circle((panel['x'], 3.5), 0.8, color='#FFB6C1', alpha=0.7)
            ax_sample.add_patch(circle)
            bbox_rect = patches.Rectangle((panel['x']-0.5, 3), 1, 1, 
                                        linewidth=2, edgecolor='red', facecolor='none')
            ax_sample.add_patch(bbox_rect)
            ax_sample.text(panel['x'], 2.5, 'R1: 0.85', ha='center', va='center', 
                          fontsize=7, color='red', fontweight='bold')
        
        elif panel['name'] == 'Attention\nHeatmap':
            # Draw heatmap representation
            circle = Circle((panel['x'], 3.5), 0.8, color='blue', alpha=0.8)
            ax_sample.add_patch(circle)
            ax_sample.text(panel['x'], 3.5, 'Heat\nMap', ha='center', va='center', 
                          fontsize=8, color='white', fontweight='bold')
        
        elif panel['name'] == 'Combined\nView':
            # Draw combined representation
            circle = Circle((panel['x'], 3.5), 0.8, color='#FFB6C1', alpha=0.5)
            ax_sample.add_patch(circle)
            circle_overlay = Circle((panel['x'], 3.5), 0.8, color='blue', alpha=0.3)
            ax_sample.add_patch(circle_overlay)
            bbox_rect = patches.Rectangle((panel['x']-0.5, 3), 1, 1, 
                                        linewidth=2, edgecolor='red', facecolor='none')
            ax_sample.add_patch(bbox_rect)
    
    ax_sample.axis('off')
    
    # ============================================================================
    # FINALIZATION
    # ============================================================================
    
    # Overall title
    fig.suptitle('MedXplain-VQA: Explainable Medical Visual Question Answering', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Figure 1 (Introduction Overview) saved to {output_path}")
    print(f"📊 Figure dimensions: 20x14 inches at 300 DPI")
    print(f"🎯 Ready for paper Section 1 (Introduction)")
    
    return output_path

def create_system_architecture_figure(output_path="paper_figures/figure2_system_architecture.png"):
    """
    Create Figure 2 for methodology section - detailed system architecture
    """
    
    fig = plt.figure(figsize=(22, 16))
    
    # This would be a more detailed technical architecture diagram
    # Showing internal components, data flows, model interactions
    
    # [Implementation for Figure 2 - detailed architecture]
    # This would show:
    # - BLIP2 internal structure
    # - Query reformulation process
    # - Grad-CAM mechanism
    # - Bounding box extraction algorithm
    # - Chain-of-thought generation
    # - LLM integration
    
    plt.suptitle('MedXplain-VQA Detailed System Architecture', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.5, 'Detailed Architecture Diagram\n(Figure 2 - for Methodology section)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 2 (System Architecture) template saved to {output_path}")
    
    return output_path

def create_performance_comparison_figure(output_path="paper_figures/figure3_performance_comparison.png"):
    """
    Create Figure 3 for results section - performance comparison
    """
    
    fig = plt.figure(figsize=(18, 12))
    
    # This would show performance comparison across different modes
    # Using real evaluation results
    
    plt.suptitle('MedXplain-VQA Performance Comparison', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.5, 'Performance Comparison Charts\n(Figure 3 - for Results section)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3 (Performance Comparison) template saved to {output_path}")
    
    return output_path

def main():
    """Generate all paper figures"""
    print("🎨 Generating Paper Figures for MedXplain-VQA")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("paper_figures", exist_ok=True)
    
    # Generate figures
    fig1_path = create_introduction_overview_figure()
    fig2_path = create_system_architecture_figure() 
    fig3_path = create_performance_comparison_figure()
    
    print("\n📋 PAPER FIGURES SUMMARY:")
    print("=" * 40)
    print("Figure 1 (Introduction): System Overview & Motivation")
    print("Figure 2 (Methodology): Detailed System Architecture") 
    print("Figure 3 (Results): Performance Comparison")
    print("\n🎯 All figures saved to paper_figures/ directory")
    print("✅ Ready for paper integration")

if __name__ == "__main__":
    main()