import torch
import numpy as np
import logging
from PIL import Image
from typing import Dict, List, Optional, Tuple

from .grad_cam import GradCAM  # Import GradCAM hiện tại
from .bounding_box_extractor import BoundingBoxExtractor

logger = logging.getLogger(__name__)

class EnhancedGradCAM:
    """
    Enhanced Grad-CAM using existing GradCAM + BoundingBoxExtractor
    Optimized for test_0001.jpg and medical images
    """
    
    def __init__(self, model, layer_name="vision_model.encoder.layers.11", 
                 bbox_config=None):
        """Initialize with existing GradCAM"""
        
        # Use existing GradCAM
        self.grad_cam = GradCAM(model, layer_name)
        
        # Initialize bounding box extractor
        self.bbox_extractor = BoundingBoxExtractor(bbox_config)
        
        logger.info("EnhancedGradCAM initialized with existing components")
    
    def analyze_image_with_question(self, image: Image.Image, question: str,
                                   save_dir: Optional[str] = None) -> Dict:
        """
        Complete analysis: GradCAM + Bounding Boxes
        
        Args:
            image: PIL Image
            question: Question string
            save_dir: Optional save directory
            
        Returns:
            Complete analysis result
        """
        logger.info(f"Analyzing image with question: '{question}'")
        
        result = {
            'success': False,
            'image_size': image.size,
            'question': question,
            'heatmap': None,
            'regions': [],
            'error': None
        }
        
        try:
            # Generate Grad-CAM heatmap using existing implementation
            logger.info("Generating Grad-CAM heatmap")
            heatmap = self.grad_cam(image, question, original_size=image.size)
            
            if heatmap is None:
                result['error'] = 'Grad-CAM generation failed'
                return result
            
            logger.info(f"Heatmap generated: {heatmap.shape}, range: {heatmap.min():.3f}-{heatmap.max():.3f}")
            result['heatmap'] = heatmap
            
            # Extract bounding boxes
            logger.info("Extracting bounding box regions")
            regions = self.bbox_extractor.extract_attention_regions(heatmap, image.size)
            result['regions'] = regions
            
            # Create visualization if save_dir provided
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                
                viz_path = os.path.join(save_dir, 'gradcam_with_bbox.png')
                fig = self.bbox_extractor.visualize_regions(image, regions, heatmap, viz_path)
                
                import matplotlib.pyplot as plt
                plt.close(fig)
                
                result['visualization_path'] = viz_path
            
            result['success'] = True
            logger.info(f"Analysis completed: {len(regions)} regions found")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            result['error'] = str(e)
        
        finally:
            # Clean up GradCAM hooks
            self.grad_cam.remove_hooks()
        
        return result
    
    def get_summary(self, analysis_result: Dict) -> Dict:
        """Get analysis summary"""
        if not analysis_result['success']:
            return {
                'status': 'failed',
                'error': analysis_result.get('error'),
                'regions_found': 0
            }
        
        regions = analysis_result['regions']
        
        if not regions:
            return {
                'status': 'no_regions',
                'regions_found': 0
            }
        
        scores = [r['attention_score'] for r in regions]
        
        return {
            'status': 'success',
            'regions_found': len(regions),
            'avg_attention': float(np.mean(scores)),
            'max_attention': float(max(scores)),
            'primary_region': {
                'bbox': regions[0]['bbox'],
                'score': regions[0]['attention_score']
            }
        }
