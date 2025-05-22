import torch
import numpy as np
import logging
from PIL import Image
from typing import Dict, List, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)

class VisualContextExtractor:
    """
    Extract visual context from medical images to ground questions in image content
    """
    
    def __init__(self, blip_model, config):
        """
        Initialize Visual Context Extractor
        
        Args:
            blip_model: BLIP model for feature extraction
            config: Configuration object
        """
        self.blip_model = blip_model
        self.config = config
        self.device = blip_model.device
        
        # Medical imaging terminology mapping
        self.anatomical_regions = {
            'chest': ['lung', 'heart', 'chest', 'thorax', 'pulmonary', 'cardiac'],
            'abdomen': ['liver', 'kidney', 'stomach', 'intestine', 'abdominal', 'hepatic'],
            'brain': ['brain', 'cerebral', 'cranial', 'neurological'],
            'bone': ['bone', 'skeletal', 'fracture', 'joint', 'orthopedic'],
            'skin': ['skin', 'dermatological', 'lesion', 'rash']
        }
        
        self.pathology_patterns = {
            'mass': ['mass', 'tumor', 'lesion', 'nodule', 'growth'],
            'inflammation': ['inflammation', 'swelling', 'edema', 'infiltrate'],
            'fracture': ['fracture', 'break', 'crack', 'displacement'],
            'infection': ['infection', 'abscess', 'pus', 'sepsis'],
            'necrosis': ['necrosis', 'death', 'necrotic', 'gangrene']
        }
        
        logger.info("Visual Context Extractor initialized")
    
    def extract_visual_features(self, image: Image.Image) -> Dict:
        """
        Extract visual features from image using BLIP vision encoder
        
        Args:
            image: PIL Image
            
        Returns:
            Dict containing visual features and metadata
        """
        try:
            # Process image through BLIP processor
            inputs = self.blip_model.processor(images=image, return_tensors="pt")
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(self.device)
            
            with torch.no_grad():
                # Extract vision features
                vision_outputs = self.blip_model.model.vision_model(inputs.pixel_values)
                
                # Get different levels of features
                features = {
                    'last_hidden_state': vision_outputs.last_hidden_state,
                    'pooler_output': getattr(vision_outputs, 'pooler_output', None)
                }
                
                # Compute feature statistics
                feature_stats = self._compute_feature_statistics(features)
                
                # Extract spatial attention patterns
                spatial_patterns = self._extract_spatial_patterns(vision_outputs.last_hidden_state)
                
                return {
                    'features': features,
                    'statistics': feature_stats,
                    'spatial_patterns': spatial_patterns,
                    'image_size': image.size
                }
                
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    def _compute_feature_statistics(self, features: Dict) -> Dict:
        """Compute statistical measures of visual features"""
        stats = {}
        
        if 'last_hidden_state' in features and features['last_hidden_state'] is not None:
            hidden_state = features['last_hidden_state'][0]  # Remove batch dim
            
            stats['feature_mean'] = torch.mean(hidden_state).item()
            stats['feature_std'] = torch.std(hidden_state).item()
            stats['feature_max'] = torch.max(hidden_state).item()
            stats['feature_min'] = torch.min(hidden_state).item()
            
            # Compute activation sparsity
            positive_activations = torch.sum(hidden_state > 0).item()
            total_activations = hidden_state.numel()
            stats['activation_sparsity'] = positive_activations / total_activations
            
        return stats
    
    def _extract_spatial_patterns(self, hidden_state: torch.Tensor) -> Dict:
        """Extract spatial attention patterns from hidden states"""
        try:
            # hidden_state shape: [batch, seq_len, hidden_dim]
            hidden_state = hidden_state[0]  # Remove batch dimension
            
            # Compute attention-like patterns
            attention_weights = torch.mean(hidden_state, dim=-1)  # [seq_len]
            
            # Convert to spatial map (assuming square grid)
            seq_len = attention_weights.shape[0]
            
            # Handle potential CLS token
            if int(np.sqrt(seq_len - 1))**2 == seq_len - 1:
                # Remove CLS token
                spatial_attention = attention_weights[1:]
                grid_size = int(np.sqrt(seq_len - 1))
            elif int(np.sqrt(seq_len))**2 == seq_len:
                spatial_attention = attention_weights
                grid_size = int(np.sqrt(seq_len))
            else:
                # Fallback
                spatial_attention = attention_weights[:196]  # 14x14
                grid_size = 14
            
            # Reshape to spatial grid
            attention_map = spatial_attention[:grid_size*grid_size].reshape(grid_size, grid_size)
            attention_map = attention_map.cpu().numpy()
            
            # Normalize
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            
            return {
                'attention_map': attention_map,
                'grid_size': grid_size,
                'attention_entropy': self._compute_attention_entropy(attention_map),
                'focus_regions': self._identify_focus_regions(attention_map)
            }
            
        except Exception as e:
            logger.error(f"Error extracting spatial patterns: {e}")
            return {}
    
    def _compute_attention_entropy(self, attention_map: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        # Flatten and normalize
        flat_attention = attention_map.flatten()
        flat_attention = flat_attention / (flat_attention.sum() + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(flat_attention * np.log(flat_attention + 1e-8))
        return float(entropy)
    
    def _identify_focus_regions(self, attention_map: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Identify high-attention regions in the attention map"""
        # Find regions above threshold
        high_attention = attention_map > threshold
        
        # Find connected components
        from scipy import ndimage
        labeled_regions, num_regions = ndimage.label(high_attention)
        
        regions = []
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions == i
            region_coords = np.where(region_mask)
            
            if len(region_coords[0]) > 0:
                # Compute bounding box
                min_row, max_row = np.min(region_coords[0]), np.max(region_coords[0])
                min_col, max_col = np.min(region_coords[1]), np.max(region_coords[1])
                
                # Compute region statistics
                region_attention = attention_map[region_mask]
                
                regions.append({
                    'bbox': [min_col, min_row, max_col - min_col + 1, max_row - min_row + 1],
                    'center': [np.mean(region_coords[1]), np.mean(region_coords[0])],
                    'mean_attention': np.mean(region_attention),
                    'max_attention': np.max(region_attention),
                    'area': len(region_coords[0])
                })
        
        # Sort by mean attention (descending)
        regions.sort(key=lambda x: x['mean_attention'], reverse=True)
        return regions
    
    def generate_visual_description(self, image: Image.Image, visual_features: Dict) -> str:
        """
        Generate natural language description of visual content
        
        Args:
            image: PIL Image
            visual_features: Extracted visual features
            
        Returns:
            String description of visual content
        """
        try:
            descriptions = []
            
            # Image characteristics
            width, height = image.size
            descriptions.append(f"Image dimensions: {width}x{height}")
            
            # Feature statistics
            if 'statistics' in visual_features:
                stats = visual_features['statistics']
                if 'activation_sparsity' in stats:
                    sparsity = stats['activation_sparsity']
                    if sparsity > 0.7:
                        descriptions.append("High visual complexity detected")
                    elif sparsity < 0.3:
                        descriptions.append("Low visual complexity detected")
            
            # Spatial patterns
            if 'spatial_patterns' in visual_features:
                patterns = visual_features['spatial_patterns']
                
                # Attention entropy
                if 'attention_entropy' in patterns:
                    entropy = patterns['attention_entropy']
                    if entropy > 2.5:
                        descriptions.append("Distributed attention across multiple regions")
                    elif entropy < 1.5:
                        descriptions.append("Focused attention on specific regions")
                
                # Focus regions
                if 'focus_regions' in patterns:
                    regions = patterns['focus_regions']
                    if len(regions) > 0:
                        primary_region = regions[0]
                        center_x, center_y = primary_region['center']
                        grid_size = patterns.get('grid_size', 14)
                        
                        # Convert to relative positions
                        rel_x = center_x / grid_size
                        rel_y = center_y / grid_size
                        
                        # Describe location
                        h_pos = "left" if rel_x < 0.33 else "right" if rel_x > 0.67 else "center"
                        v_pos = "upper" if rel_y < 0.33 else "lower" if rel_y > 0.67 else "middle"
                        
                        descriptions.append(f"Primary focus in {v_pos} {h_pos} region")
                        
                        if len(regions) > 1:
                            descriptions.append(f"Secondary attention regions detected ({len(regions)-1} additional)")
            
            return "; ".join(descriptions)
            
        except Exception as e:
            logger.error(f"Error generating visual description: {e}")
            return "Visual analysis unavailable"
    
    def identify_anatomical_context(self, visual_description: str, question: str) -> str:
        """
        Identify anatomical context from visual features and question
        
        Args:
            visual_description: Generated visual description
            question: Original question text
            
        Returns:
            Identified anatomical context
        """
        # Combine visual description and question for analysis
        combined_text = f"{visual_description} {question}".lower()
        
        # Check for anatomical regions
        detected_regions = []
        for region, keywords in self.anatomical_regions.items():
            if any(keyword in combined_text for keyword in keywords):
                detected_regions.append(region)
        
        # Check for pathology patterns
        detected_pathology = []
        for pathology, keywords in self.pathology_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                detected_pathology.append(pathology)
        
        # Construct context
        context_parts = []
        if detected_regions:
            context_parts.append(f"anatomical context: {', '.join(detected_regions)}")
        if detected_pathology:
            context_parts.append(f"pathology indicators: {', '.join(detected_pathology)}")
        
        return "; ".join(context_parts) if context_parts else "general medical imaging context"
    
    def extract_complete_context(self, image: Image.Image, question: str) -> Dict:
        """
        Extract complete visual context for question reformulation
        
        Args:
            image: PIL Image
            question: Original question
            
        Returns:
            Complete context dictionary
        """
        logger.info("Extracting complete visual context")
        
        # Extract visual features
        visual_features = self.extract_visual_features(image)
        
        # Generate visual description
        visual_description = self.generate_visual_description(image, visual_features)
        
        # Identify anatomical context
        anatomical_context = self.identify_anatomical_context(visual_description, question)
        
        # Compile complete context
        complete_context = {
            'visual_features': visual_features,
            'visual_description': visual_description,
            'anatomical_context': anatomical_context,
            'question': question,
            'image_metadata': {
                'size': image.size,
                'mode': image.mode
            }
        }
        
        logger.info(f"Visual context extracted: {visual_description}")
        return complete_context
