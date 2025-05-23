import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

logger = logging.getLogger(__name__)

class SimpleBoundingBoxExtractor:
    """
    Simple and Reliable Bounding Box Extractor
    No sklearn dependency, focused on core functionality
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Parameters
        self.attention_threshold = self.config.get('attention_threshold', 0.3)
        self.min_region_size = self.config.get('min_region_size', 10)
        self.max_regions = self.config.get('max_regions', 5)
        self.box_expansion = self.config.get('box_expansion', 0.1)
        
        logger.info(f"SimpleBoundingBoxExtractor initialized with threshold={self.attention_threshold}")
    
    def extract_regions_from_heatmap(self, heatmap: np.ndarray, 
                                   image_size: Tuple[int, int] = (224, 224)) -> List[Dict]:
        """
        Extract bounding box regions from Grad-CAM heatmap
        
        Args:
            heatmap: Grad-CAM attention heatmap (H, W)
            image_size: Target image size (width, height)
            
        Returns:
            List of region dictionaries with bounding boxes
        """
        if heatmap is None or heatmap.size == 0:
            logger.warning("Empty or None heatmap provided")
            return []
        
        logger.info(f"Processing heatmap: {heatmap.shape}, target size: {image_size}")
        
        try:
            # 1. Normalize heatmap
            heatmap_norm = self._normalize_heatmap(heatmap)
            logger.debug(f"Normalized heatmap range: {heatmap_norm.min():.3f} - {heatmap_norm.max():.3f}")
            
            # 2. Create binary mask using threshold
            binary_mask = heatmap_norm > self.attention_threshold
            logger.debug(f"Binary mask pixels above threshold: {np.sum(binary_mask)}")
            
            if np.sum(binary_mask) == 0:
                logger.warning("No pixels above threshold, lowering threshold")
                # Try with lower threshold
                binary_mask = heatmap_norm > (self.attention_threshold * 0.5)
                
            if np.sum(binary_mask) == 0:
                logger.warning("Still no regions found, returning empty list")
                return []
            
            # 3. Find connected components
            regions = self._extract_connected_components(binary_mask, heatmap_norm, image_size)
            
            # 4. Post-process regions
            final_regions = self._post_process_regions(regions, image_size)
            
            logger.info(f"Successfully extracted {len(final_regions)} regions")
            return final_regions
            
        except Exception as e:
            logger.error(f"Error extracting regions: {e}")
            return []
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1] range"""
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            return (heatmap - hmin) / (hmax - hmin)
        else:
            return np.zeros_like(heatmap)
    
    def _extract_connected_components(self, binary_mask: np.ndarray,
                                    heatmap: np.ndarray,
                                    image_size: Tuple[int, int]) -> List[Dict]:
        """Extract connected components and convert to bounding boxes"""
        # Label connected components
        labeled_mask, num_components = ndimage.label(binary_mask)
        logger.debug(f"Found {num_components} connected components")
        
        regions = []
        for i in range(1, num_components + 1):
            component_mask = labeled_mask == i
            component_coords = np.where(component_mask)
            
            # Filter small regions
            if len(component_coords[0]) < self.min_region_size:
                continue
            
            # Get bounding box coordinates in heatmap space
            min_row, max_row = np.min(component_coords[0]), np.max(component_coords[0])
            min_col, max_col = np.min(component_coords[1]), np.max(component_coords[1])
            
            # Convert to image coordinates
            scale_x = image_size[0] / heatmap.shape[1]  # width scale
            scale_y = image_size[1] / heatmap.shape[0]  # height scale
            
            bbox = [
                int(min_col * scale_x),  # x
                int(min_row * scale_y),  # y  
                int((max_col - min_col + 1) * scale_x),  # width
                int((max_row - min_row + 1) * scale_y)   # height
            ]
            
            # Calculate region statistics
            region_attention = heatmap[component_mask]
            
            region = {
                'bbox': bbox,
                'center': [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2],
                'area': len(component_coords[0]),
                'attention_score': float(np.mean(region_attention)),
                'max_attention': float(np.max(region_attention)),
                'confidence': float(np.mean(region_attention) * 0.8 + np.max(region_attention) * 0.2)
            }
            
            regions.append(region)
            logger.debug(f"Region {i}: bbox={bbox}, score={region['attention_score']:.3f}")
        
        return regions
    
    def _post_process_regions(self, regions: List[Dict], 
                            image_size: Tuple[int, int]) -> List[Dict]:
        """Post-process regions: sort, limit, expand boxes"""
        if not regions:
            return regions
        
        # Sort by attention score (highest first)
        sorted_regions = sorted(regions, key=lambda x: x['attention_score'], reverse=True)
        
        # Limit number of regions
        limited_regions = sorted_regions[:self.max_regions]
        
        # Add rank and expand bounding boxes
        for i, region in enumerate(limited_regions):
            region['rank'] = i + 1
            region['bbox'] = self._expand_bbox(region['bbox'], image_size)
        
        return limited_regions
    
    def _expand_bbox(self, bbox: List[int], image_size: Tuple[int, int]) -> List[int]:
        """Expand bounding box for better visualization"""
        x, y, w, h = bbox
        
        # Calculate expansion
        exp_w = int(w * self.box_expansion)
        exp_h = int(h * self.box_expansion)
        
        # Apply expansion with bounds checking
        new_x = max(0, x - exp_w)
        new_y = max(0, y - exp_h)
        new_w = min(image_size[0] - new_x, w + 2 * exp_w)
        new_h = min(image_size[1] - new_y, h + 2 * exp_h)
        
        return [new_x, new_y, new_w, new_h]
    
    def visualize_regions(self, image: Image.Image, regions: List[Dict],
                         heatmap: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of regions on image"""
        
        if heatmap is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image with boxes
            axes[0].imshow(image)
            axes[0].set_title(f'Image with Bounding Boxes ({len(regions)} regions)')
            axes[0].axis('off')
            self._draw_boxes_on_axis(axes[0], regions)
            
            # Heatmap
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Combined view
            axes[2].imshow(image, alpha=0.7)
            axes[2].imshow(heatmap, cmap='jet', alpha=0.4)
            axes[2].set_title('Combined View')
            axes[2].axis('off')
            self._draw_boxes_on_axis(axes[2], regions)
            
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Image with boxes
            axes[0].imshow(image)
            axes[0].set_title(f'Image with Bounding Boxes ({len(regions)} regions)')
            axes[0].axis('off')
            self._draw_boxes_on_axis(axes[0], regions)
            
            # Region info
            self._draw_region_info(axes[1], regions)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def _draw_boxes_on_axis(self, ax, regions: List[Dict]):
        """Draw bounding boxes on matplotlib axis"""
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
        
        for i, region in enumerate(regions):
            bbox = region['bbox']
            color = colors[i % len(colors)]
            
            # Draw rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                bbox[0], bbox[1] - 5,
                f"R{region['rank']}: {region['attention_score']:.3f}",
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
    
    def _draw_region_info(self, ax, regions: List[Dict]):
        """Draw region information text"""
        ax.axis('off')
        
        if not regions:
            ax.text(0.5, 0.5, 'No regions detected', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return
        
        info_lines = ['Region Statistics:', '']
        for region in regions:
            line = (f"Region {region['rank']}: "
                   f"Score={region['attention_score']:.3f}, "
                   f"Confidence={region['confidence']:.3f}")
            info_lines.append(line)
            
            bbox_line = f"  BBox: {region['bbox']}"
            info_lines.append(bbox_line)
            info_lines.append('')
        
        ax.text(0.05, 0.95, '\n'.join(info_lines),
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Region Details')
