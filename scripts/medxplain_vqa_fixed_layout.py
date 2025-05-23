#!/usr/bin/env python
# 🆕 ENHANCED LAYOUT: Fixed visualization with separate original image + reasoning chain display

# [Previous imports and functions remain the same until create_visualization...]

def create_visualization(result, output_dir, logger):
    """
    🆕 ENHANCED LAYOUT: Create visualization với separate original image + reasoning chain
    """
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    mode = result['mode']
    image = result['image']
    sample_id = Path(result['image_path']).stem
    success = result['success']
    bbox_enabled = result.get('bbox_enabled', False)
    bbox_regions = result.get('bbox_regions', [])
    
    try:
        if mode == 'basic_vqa':
            # Basic visualization (2x1 layout)
            fig = plt.figure(figsize=(12, 6))
            
            # Image
            ax_image = plt.subplot(1, 2, 1)
            ax_image.imshow(image)
            ax_image.set_title(f"MedXplain-VQA: {sample_id}", fontsize=12)
            ax_image.axis('off')
            
            # Text
            ax_text = plt.subplot(1, 2, 2)
            text_content = (
                f"Question: {result['question']}\n\n"
                f"Ground truth: {result['ground_truth']}\n\n"
                f"MedXplain-VQA answer: {result['unified_answer']}"
            )
            
            if not success:
                text_content += f"\n\nErrors: {'; '.join(result['error_messages'])}"
            
            ax_text.text(0.01, 0.99, text_content, transform=ax_text.transAxes,
                        fontsize=10, verticalalignment='top', wrap=True)
            ax_text.axis('off')
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, f"medxplain_basic_{sample_id}.png")
            
        else:  # explainable_vqa mode
            # 🆕 ENHANCED LAYOUT: 3x2 grid layout
            enable_cot = result['chain_of_thought_enabled']
            
            if enable_cot:
                # 3x2 layout: Row 1: Original | Heatmap | Combined
                #             Row 2: Reasoning Chain (spans full width)
                fig = plt.figure(figsize=(20, 14))
                
                # ROW 1: Images
                # Original image (clean, no bounding boxes)
                ax_orig = plt.subplot2grid((3, 3), (0, 0))
                ax_orig.imshow(image)
                ax_orig.set_title("🖼️ Original Medical Image", fontsize=12, fontweight='bold')
                ax_orig.axis('off')
                
                # Grad-CAM heatmap
                ax_heatmap = plt.subplot2grid((3, 3), (0, 1))
                if result['grad_cam_heatmap'] is not None:
                    ax_heatmap.imshow(result['grad_cam_heatmap'], cmap='jet')
                    mode_label = "Enhanced" if bbox_enabled else "Basic"
                    ax_heatmap.set_title(f"🎯 {mode_label} Attention Heatmap", fontsize=12, fontweight='bold')
                else:
                    ax_heatmap.text(0.5, 0.5, "Heatmap not available", ha='center', va='center')
                    ax_heatmap.set_title("🎯 Attention Heatmap (N/A)", fontsize=12)
                ax_heatmap.axis('off')
                
                # Combined view with bounding boxes
                ax_combined = plt.subplot2grid((3, 3), (0, 2))
                ax_combined.imshow(image, alpha=0.7)
                if result['grad_cam_heatmap'] is not None:
                    ax_combined.imshow(result['grad_cam_heatmap'], cmap='jet', alpha=0.4)
                
                # 🆕 ENHANCED: Draw bounding boxes on combined view
                if bbox_regions:
                    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
                    for i, region in enumerate(bbox_regions[:5]):
                        bbox = region['bbox']
                        color = colors[i % len(colors)]
                        score = region.get('attention_score', region.get('score', 0))
                        
                        # Draw bounding box
                        rect = patches.Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=3, edgecolor=color, facecolor='none', alpha=0.9
                        )
                        ax_combined.add_patch(rect)
                        
                        # Add label
                        ax_combined.text(
                            bbox[0], bbox[1] - 8,
                            f"R{i+1}: {score:.3f}",
                            color=color, fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9)
                        )
                    
                    ax_combined.set_title(f"🎯 Combined: Image + Attention + {len(bbox_regions)} Bounding Boxes", fontsize=12, fontweight='bold')
                else:
                    ax_combined.set_title("🎯 Combined: Image + Attention", fontsize=12, fontweight='bold')
                ax_combined.axis('off')
                
                # ROW 2: Question, Answer, Ground Truth
                ax_qa = plt.subplot2grid((3, 3), (1, 0), colspan=3)
                qa_content = (
                    f"❓ QUESTION: {result['question']}\n\n"
                    f"🔄 REFORMULATED: {result['reformulated_question']}\n\n"
                    f"✅ GROUND TRUTH: {result['ground_truth']}\n\n" 
                    f"🤖 MEDXPLAIN-VQA ANSWER: {result['unified_answer']}"
                )
                
                ax_qa.text(0.01, 0.99, qa_content, transform=ax_qa.transAxes,
                          fontsize=11, verticalalignment='top', wrap=True, fontweight='normal')
                ax_qa.set_title("📋 Question-Answer Analysis", fontsize=12, fontweight='bold')
                ax_qa.axis('off')
                
                # ROW 3: Chain-of-Thought Reasoning (Full Width)
                ax_reasoning = plt.subplot2grid((3, 3), (2, 0), colspan=3)
                
                if result['reasoning_result'] and result['reasoning_result']['success']:
                    reasoning_chain = result['reasoning_result']['reasoning_chain']
                    steps = reasoning_chain['steps']
                    confidence = reasoning_chain['overall_confidence']
                    
                    reasoning_text = f"🧠 CHAIN-OF-THOUGHT REASONING (Confidence: {confidence:.3f})\n"
                    reasoning_text += f"Flow: {reasoning_chain['flow_type']} | Steps: {len(steps)}\n\n"
                    
                    # Show all reasoning steps with better formatting
                    for i, step in enumerate(steps):
                        step_confidence = step.get('confidence', 0.0)
                        step_content = step['content'][:150] + "..." if len(step['content']) > 150 else step['content']
                        reasoning_text += f"{i+1}. {step['type'].upper().replace('_', ' ')} (conf: {step_confidence:.2f}):\n"
                        reasoning_text += f"   {step_content}\n\n"
                else:
                    reasoning_text = "🧠 CHAIN-OF-THOUGHT REASONING: Not available or failed"
                    if result.get('reasoning_result') and not result['reasoning_result']['success']:
                        reasoning_text += f"\nError: {result['reasoning_result'].get('error', 'Unknown')}"
                
                ax_reasoning.text(0.01, 0.99, reasoning_text, transform=ax_reasoning.transAxes,
                                fontsize=10, verticalalignment='top', wrap=True, fontfamily='monospace')
                ax_reasoning.set_title("🧠 Detailed Reasoning Chain", fontsize=12, fontweight='bold')
                ax_reasoning.axis('off')
                
            else:
                # 2x2 layout for basic explainable (no Chain-of-Thought)
                fig = plt.figure(figsize=(16, 12))
                
                # Original image 
                ax_image = plt.subplot2grid((2, 2), (0, 0))
                ax_image.imshow(image)
                ax_image.set_title("🖼️ Original Medical Image", fontsize=12, fontweight='bold')
                ax_image.axis('off')
                
                # Grad-CAM with bounding boxes
                ax_heatmap = plt.subplot2grid((2, 2), (0, 1))
                if result['grad_cam_heatmap'] is not None:
                    ax_heatmap.imshow(result['grad_cam_heatmap'], cmap='jet')
                    
                    # Add bounding boxes to heatmap view
                    if bbox_regions:
                        colors = ['white', 'yellow', 'cyan', 'magenta', 'lime']
                        for i, region in enumerate(bbox_regions[:5]):
                            bbox = region['bbox']
                            color = colors[i % len(colors)]
                            score = region.get('attention_score', region.get('score', 0))
                            
                            rect = patches.Rectangle(
                                (bbox[0], bbox[1]), bbox[2], bbox[3],
                                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
                            )
                            ax_heatmap.add_patch(rect)
                            
                            ax_heatmap.text(
                                bbox[0], bbox[1] - 5,
                                f"R{i+1}: {score:.3f}",
                                color=color, fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7)
                            )
                    
                    mode_label = "Enhanced" if bbox_enabled else "Basic"
                    bbox_info = f" + {len(bbox_regions)} Boxes" if bbox_regions else ""
                    ax_heatmap.set_title(f"🎯 {mode_label} Heatmap{bbox_info}", fontsize=12, fontweight='bold')
                else:
                    ax_heatmap.text(0.5, 0.5, "Heatmap not available", ha='center', va='center')
                    ax_heatmap.set_title("🎯 Attention Heatmap (N/A)", fontsize=12)
                ax_heatmap.axis('off')
                
                # Question-Answer area (full width bottom)
                ax_text = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            
            # Common enhanced text content for explainable mode (bottom section)
            if not enable_cot:  # Only for basic explainable mode
                text_content = (
                    f"❓ QUESTION: {result['question']}\n\n"
                    f"🔄 REFORMULATED: {result['reformulated_question']}\n\n"
                    f"✅ GROUND TRUTH: {result['ground_truth']}\n\n"
                    f"🤖 MEDXPLAIN-VQA ANSWER: {result['unified_answer']}\n\n"
                    f"🔄 PROCESSING: {' → '.join(result['processing_steps'])}\n"
                    f"📊 REFORMULATION QUALITY: {result['reformulation_quality']:.3f}"
                )
                
                # Add bounding box information
                if bbox_regions:
                    text_content += f" | 🎯 BOUNDING BOXES: {len(bbox_regions)} detected"
                    avg_score = sum(r.get('attention_score', r.get('score', 0)) for r in bbox_regions) / len(bbox_regions)
                    text_content += f" (avg score: {avg_score:.3f})"
                
                # Add error information if any
                if result['error_messages']:
                    text_content += f"\n\n⚠️ ISSUES: {'; '.join(result['error_messages'])}"
                
                ax_text.text(0.01, 0.99, text_content, transform=ax_text.transAxes,
                            fontsize=10, verticalalignment='top', wrap=True)
                ax_text.axis('off')
            
            # Set overall title
            mode_title = "Enhanced" if enable_cot else "Basic"
            bbox_status = f"+ BBox" if bbox_enabled else ""
            success_indicator = "✅ SUCCESS" if success else "⚠️ WARNING"
            plt.suptitle(f"{success_indicator} MedXplain-VQA {mode_title} {bbox_status} Explainable Analysis: {sample_id}", 
                        fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            mode_suffix = "enhanced" if enable_cot else "explainable"
            bbox_suffix = "_bbox" if bbox_enabled else ""
            output_file = os.path.join(output_dir, f"medxplain_{mode_suffix}{bbox_suffix}_{sample_id}.png")
        
        # Save visualization
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.5, dpi=150)
        plt.close(fig)
        logger.info(f"✅ Enhanced layout visualization saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error creating enhanced layout visualization: {e}")
        return None

# [Rest of the file remains the same...]
