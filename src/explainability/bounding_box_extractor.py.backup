import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

logger = logging.getLogger(__name__)

class BoundingBoxExtractor:
    """
    Simple Bounding Box Extractor for Grad-CAM attention regions
    Optimized for test_0001.jpg and similar medical images
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Simple parameters
        self.attention_threshold = self.config.get('attention_threshold', 0.3)
        self.min_region_size = self.config.get('min_region_size', 8)
        self.max_regions = self.config.get('max_regions', 5)
        self.box_expansion = self.config.get('box_expansion', 0.1)
        
        logger.info(f"BoundingBoxExtractor initialized (threshold={self.attention_threshold})")
    
    def extract_attention_regions(self, heatmap: np.ndarray, 
                                 image_size: Tuple[int, int]) -> List[Dict]:
        """
        Extract bounding box regions from Grad-CAM heatmap
        
        Args:
            heatmap: Grad-CAM attention heatmap (H, W)
            image_size: Target image size (width, height)
            
        Returns:
            List of region dictionaries with bounding boxes
        """
        if heatmap is None or heatmap.size == 0:
            logger.warning("Empty heatmap provided")
            return []
        
        logger.info(f"Extracting regions from heatmap: {heatmap.shape} -> {image_size}")
        
        try:
            # Normalize heatmap
            heatmap_norm = self._normalize_heatmap(heatmap)
            
            # Create binary mask
            binary_mask = heatmap_norm > self.attention_threshold
            
            # If no regions found, try lower threshold
            if np.sum(binary_mask) == 0:
                binary_mask = heatmap_norm > (self.attention_threshold * 0.6)
                logger.info("Using lower threshold")
            
            if np.sum(binary_mask) == 0:
                return []
            
            # Find connected components
            regions = self._extract_connected_components(binary_mask, heatmap_norm, image_size)
            
            # Post-process
            final_regions = self._post_process_regions(regions, image_size)
            
            logger.info(f"Extracted {len(final_regions)} regions")
            return final_regions
            
        except Exception as e:
            logger.error(f"Error extracting regions: {e}")
            return []
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1]"""
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            return (heatmap - hmin) / (hmax - hmin)
        return np.zeros_like(heatmap)
    
    def _extract_connected_components(self, binary_mask: np.ndarray,
                                    heatmap: np.ndarray,
                                    image_size: Tuple[int, int]) -> List[Dict]:
        """Extract connected components as bounding boxes"""
        labeled_mask, num_components = ndimage.label(binary_mask)
        
        regions = []
        for i in range(1, num_components + 1):
            component_mask = labeled_mask == i
            component_coords = np.where(component_mask)
            
            if len(component_coords[0]) < self.min_region_size:
                continue
            
            # Get bounding box in heatmap coordinates
            min_row, max_row = np.min(component_coords[0]), np.max(component_coords[0])
            min_col, max_col = np.min(component_coords[1]), np.max(component_coords[1])
            
            # Scale to image coordinates
            scale_x = image_size[0] / heatmap.shape[1]
            scale_y = image_size[1] / heatmap.shape[0]
            
            bbox = [
                int(min_col * scale_x),
                int(min_row * scale_y),
                int((max_col - min_col + 1) * scale_x),
                int((max_row - min_row + 1) * scale_y)
            ]
            
            # Calculate statistics
            region_attention = heatmap[component_mask]
            
            region = {
                'bbox': bbox,
                'center': [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2],
                'attention_score': float(np.mean(region_attention)),
                'max_attention': float(np.max(region_attention)),
                'confidence': float(np.mean(region_attention) * 0.9 + np.max(region_attention) * 0.1)
            }
            
            regions.append(region)
        
        return regions
    
    def _post_process_regions(self, regions: List[Dict], 
                            image_size: Tuple[int, int]) -> List[Dict]:
        """Sort, limit, and refine regions"""
        if not regions:
            return regions
        
        # Sort by attention score
        sorted_regions = sorted(regions, key=lambda x: x['attention_score'], reverse=True)
        
        # Limit regions
        limited_regions = sorted_regions[:self.max_regions]
        
        # Add rank and expand boxes
        for i, region in enumerate(limited_regions):
            region['rank'] = i + 1
            region['bbox'] = self._expand_bbox(region['bbox'], image_size)
        
        return limited_regions
    
    def _expand_bbox(self, bbox: List[int], image_size: Tuple[int, int]) -> List[int]:
        """Expand bounding box slightly"""
        x, y, w, h = bbox
        
        exp_w = int(w * self.box_expansion)
        exp_h = int(h * self.box_expansion)
        
        new_x = max(0, x - exp_w)
        new_y = max(0, y - exp_h)
        new_w = min(image_size[0] - new_x, w + 2 * exp_w)
        new_h = min(image_size[1] - new_y, h + 2 * exp_h)
        
        return [new_x, new_y, new_w, new_h]
    
    def visualize_regions(self, image: Image.Image, regions: List[Dict],
                     heatmap: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
        """Create 4-panel visualization: Original | Bounding | Heatmap | Combined"""
        
        if heatmap is not None:
            # 🆕 NEW: 1x4 layout as requested
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            
            # 1. Original Image (clean)
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 2. Image + Bounding Boxes (no heatmap)
            axes[1].imshow(image)
            axes[1].set_title(f'Bounding Boxes ({len(regions)} regions)')
            axes[1].axis('off')
            self._draw_boxes(axes[1], regions)
            
            # 3. Heatmap Only
            axes[2].imshow(heatmap, cmap='jet')
            axes[2].set_title('Heatmap Only')
            axes[2].axis('off')
            
            # 4. Combined View (image + heatmap + bounding boxes)
            axes[3].imshow(image, alpha=0.6)
            axes[3].imshow(heatmap, cmap='jet', alpha=0.4)
            axes[3].set_title('Combined View')
            axes[3].axis('off')
            self._draw_boxes(axes[3], regions)
            
        else:
            # Fallback: 2 panels when no heatmap
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Image with boxes
            axes[0].imshow(image)
            axes[0].set_title(f'Image with Bounding Boxes ({len(regions)} regions)')
            axes[0].axis('off')
            self._draw_boxes(axes[0], regions)
            
            # Region info
            axes[1].axis('off')
            self._draw_region_info(axes[1], regions)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 4-panel visualization: {save_path}")
        
        return fig
    
    def _draw_boxes(self, ax, regions: List[Dict]):
        """Draw bounding boxes on axis"""
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        for i, region in enumerate(regions):
            bbox = region['bbox']
            color = colors[i % len(colors)]
            
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            ax.text(
                bbox[0], bbox[1] - 5,
                f"R{region['rank']}: {region['attention_score']:.3f}",
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
    
    def _draw_region_info(self, ax, regions: List[Dict]):
        """Draw region information"""
        if not regions:
            ax.text(0.5, 0.5, 'No regions detected', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return
        
        info = ['Region Statistics:', '']
        for region in regions:
            info.append(f"Region {region['rank']}: Score={region['attention_score']:.3f}")
            info.append(f"  BBox: {region['bbox']}")
            info.append('')
        
        ax.text(0.05, 0.95, '\n'.join(info),
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Region Details')
