import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from PIL import Image
from typing import Dict, List, Optional, Tuple
from .grad_cam import GradCAM
from .bounding_box_extractor import BoundingBoxExtractor

logger = logging.getLogger(__name__)

class EnhancedGradCAMFixed(GradCAM):
    """
    FIXED: Enhanced Grad-CAM với proper image visualization
    """
    
    def __init__(self, model, layer_name="vision_model.encoder.layers.11", 
                 bbox_config=None):
        super().__init__(model, layer_name)
        self.bbox_extractor = BoundingBoxExtractor(bbox_config)
        logger.info("Enhanced Grad-CAM FIXED initialized")
    
    def generate_complete_analysis(self, image: Image.Image, question: str,
                                  inputs: Optional[Dict] = None,
                                  original_size: Optional[Tuple[int, int]] = None,
                                  extraction_method: str = 'adaptive') -> Dict:
        """Generate complete analysis with FIXED visualization"""
        logger.info("Generating complete Grad-CAM analysis (FIXED VERSION)")
        
        # Store original image for visualization
        self.original_image = image.copy()
        
        # Generate base Grad-CAM heatmap
        heatmap = self(image, question, inputs, original_size)
        
        if heatmap is None:
            logger.error("Failed to generate Grad-CAM heatmap")
            return {
                'success': False,
                'error': 'Grad-CAM generation failed',
                'heatmap': None,
                'regions': [],
                'bounding_boxes': []
            }
        
        # Determine image size
        if original_size is None:
            original_size = image.size  # (width, height)
        
        logger.info(f"Processing image size: {original_size}")
        logger.info(f"Heatmap shape: {heatmap.shape}")
        
        # Extract attention regions and bounding boxes
        try:
            regions = self.bbox_extractor.extract_attention_regions(
                heatmap, original_size, extraction_method
            )
            
            # Generate region descriptions
            region_descriptions = self.bbox_extractor.generate_region_descriptions(
                regions, original_size
            )
            
            # Create analysis result
            analysis_result = {
                'success': True,
                'heatmap': heatmap,
                'original_image': image,  # FIXED: Store original image
                'regions': regions,
                'bounding_boxes': [region['bbox'] for region in regions],
                'region_descriptions': region_descriptions,
                'extraction_method': extraction_method,
                'total_regions': len(regions),
                'image_size': original_size,
                'average_attention': float(np.mean([r['attention_score'] for r in regions])) if regions else 0.0,
                'max_attention': float(max([r['attention_score'] for r in regions])) if regions else 0.0
            }
            
            logger.info(f"Complete analysis generated: {len(regions)} regions extracted")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in bounding box extraction: {e}")
            return {
                'success': False,
                'error': f'Bounding box extraction failed: {str(e)}',
                'heatmap': heatmap,
                'original_image': image,
                'regions': [],
                'bounding_boxes': []
            }
    
    def visualize_complete_analysis_fixed(self, analysis_result: Dict,
                                        save_path: Optional[str] = None) -> Optional[str]:
        """FIXED: Create proper visualization with original image"""
        if not analysis_result['success']:
            logger.error("Cannot visualize failed analysis")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            image = analysis_result['original_image']
            regions = analysis_result['regions']
            heatmap = analysis_result['heatmap']
            
            # Create figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Original image with bounding boxes
            ax1.imshow(image)
            ax1.set_title(f'Original Image with Bounding Boxes ({len(regions)} regions)')
            ax1.axis('off')
            
            # Add bounding boxes to original image
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
            for i, region in enumerate(regions):
                bbox = region['bbox']
                color = colors[i % len(colors)]
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=3, edgecolor=color, facecolor='none', alpha=0.9
                )
                ax1.add_patch(rect)
                
                # Add label
                ax1.text(
                    bbox[0], bbox[1] - 10,
                    f"R{region['rank']}: {region['attention_score']:.3f}",
                    color=color, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                )
            
            # 2. Heatmap only
            ax2.imshow(heatmap, cmap='jet')
            ax2.set_title('Grad-CAM Attention Heatmap')
            ax2.axis('off')
            
            # 3. Combined overlay
            ax3.imshow(image, alpha=0.6)
            ax3.imshow(heatmap, cmap='jet', alpha=0.4)
            ax3.set_title('Combined: Image + Heatmap + Bounding Boxes')
            ax3.axis('off')
            
            # Add bounding boxes to combined view
            for i, region in enumerate(regions):
                bbox = region['bbox']
                color = colors[i % len(colors)]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
                )
                ax3.add_patch(rect)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"FIXED visualization saved to {save_path}")
            
            plt.close(fig)
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating FIXED visualization: {e}")
            return None
