import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class EvidenceLinker:
    """
    🆕 ENHANCED: Links visual evidence from Grad-CAM + Bounding Boxes to reasoning steps
    Creates evidence citations for chain-of-thought reasoning with spatial bounding box support
    """
    
    def __init__(self, config):
        """
        Initialize Evidence Linker
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Evidence strength thresholds
        self.attention_thresholds = {
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4
        }
        
        # 🆕 NEW: Bounding box specific thresholds
        self.bbox_thresholds = {
            'high_confidence': 0.7,
            'medium_confidence': 0.5,
            'low_confidence': 0.3
        }
        
        # Evidence types and their characteristics
        self.evidence_types = {
            'visual_attention': {
                'description': 'Model attention focus on specific image regions',
                'strength_indicator': 'attention_score',
                'reliability': 'high'
            },
            'spatial_correlation': {
                'description': 'Spatial relationship between attention and pathology',
                'strength_indicator': 'spatial_overlap',
                'reliability': 'moderate'
            },
            'feature_correspondence': {
                'description': 'Visual features matching clinical descriptions',
                'strength_indicator': 'feature_match_score',
                'reliability': 'high'
            },
            'pattern_recognition': {
                'description': 'Recognition of known pathological patterns',
                'strength_indicator': 'pattern_confidence',
                'reliability': 'moderate'
            },
            # 🆕 NEW: Bounding box evidence types
            'bounding_box_attention': {
                'description': 'Precise spatial attention regions with bounding boxes',
                'strength_indicator': 'bbox_attention_score',
                'reliability': 'very_high'
            },
            'spatial_localization': {
                'description': 'Accurate spatial localization of pathological features',
                'strength_indicator': 'localization_precision',
                'reliability': 'high'
            }
        }
        
        logger.info("🆕 Enhanced Evidence Linker initialized with bounding box support")
    
    def extract_visual_evidence(self, image: Image.Image, 
                               grad_cam_data: Dict, 
                               visual_context: Dict) -> Dict:
        """
        🆕 ENHANCED: Extract visual evidence from image, attention data, and bounding boxes
        
        Args:
            image: PIL Image
            grad_cam_data: Grad-CAM attention data including heatmap and bounding box regions
            visual_context: Visual context from VisualContextExtractor
            
        Returns:
            Enhanced visual evidence dictionary with bounding box support
        """
        evidence = {
            'image_metadata': {
                'size': image.size,
                'mode': image.mode
            },
            'attention_evidence': {},
            'spatial_evidence': {},
            'feature_evidence': {},
            # 🆕 NEW: Bounding box evidence
            'bounding_box_evidence': {},
            'summary': {}
        }
        
        try:
            # 🆕 ENHANCED: Extract bounding box evidence first (highest priority)
            if 'regions' in grad_cam_data and grad_cam_data['regions']:
                bbox_enabled = grad_cam_data.get('bbox_enabled', False)
                
                if bbox_enabled:
                    logger.debug("🆕 Extracting enhanced bounding box evidence...")
                    evidence['bounding_box_evidence'] = self._extract_bounding_box_evidence(
                        grad_cam_data['regions'], image.size
                    )
                else:
                    logger.debug("Extracting basic attention evidence...")
                    evidence['attention_evidence'] = self._extract_attention_evidence(
                        grad_cam_data['regions'], image.size
                    )
            
            # Extract spatial evidence
            if 'spatial_patterns' in visual_context:
                evidence['spatial_evidence'] = self._extract_spatial_evidence(
                    visual_context['spatial_patterns'], image.size
                )
            
            # Extract feature evidence
            evidence['feature_evidence'] = self._extract_feature_evidence(
                visual_context.get('visual_description', ''),
                visual_context.get('anatomical_context', '')
            )
            
            # Create enhanced evidence summary
            evidence['summary'] = self._create_evidence_summary(evidence)
            
            logger.info("🆕 Enhanced visual evidence extracted successfully")
            
        except Exception as e:
            logger.error(f"Error extracting enhanced visual evidence: {e}")
            evidence['error'] = str(e)
        
        return evidence
    
    def _extract_bounding_box_evidence(self, bbox_regions: List[Dict], 
                                      image_size: Tuple[int, int]) -> Dict:
        """
        🆕 NEW: Extract evidence from bounding box regions
        
        Args:
            bbox_regions: List of bounding box region dictionaries
            image_size: (width, height) of original image
            
        Returns:
            Bounding box evidence dictionary
        """
        bbox_evidence = {
            'primary_regions': [],
            'secondary_regions': [],
            'spatial_distribution': {},
            'localization_precision': {},
            'region_characteristics': {}
        }
        
        # Sort regions by attention score
        sorted_regions = sorted(bbox_regions, key=lambda x: x.get('attention_score', x.get('score', 0)), reverse=True)
        
        # 🆕 ENHANCED: Categorize regions by bounding box confidence
        for i, region in enumerate(sorted_regions):
            score = region.get('attention_score', region.get('score', 0))
            bbox = region.get('bbox', [0, 0, 0, 0])
            
            # Calculate enhanced region info
            region_info = {
                'rank': i + 1,
                'bbox': bbox,
                'center': self._calculate_region_center(bbox),
                'score': score,
                'attention_score': score,
                'relative_size': self._calculate_bbox_relative_size(bbox, image_size),
                'strength': self._categorize_bbox_attention_strength(score),
                'spatial_location': self._describe_spatial_location(
                    self._calculate_region_center(bbox), image_size
                ),
                'region_extent': self._describe_bbox_extent(bbox, image_size)
            }
            
            # Enhanced categorization based on confidence
            if score >= self.bbox_thresholds['high_confidence']:
                bbox_evidence['primary_regions'].append(region_info)
            elif score >= self.bbox_thresholds['medium_confidence']:
                bbox_evidence['secondary_regions'].append(region_info)
        
        # 🆕 ENHANCED: Calculate spatial distribution metrics
        if sorted_regions:
            bbox_evidence['spatial_distribution'] = {
                'total_regions': len(sorted_regions),
                'high_confidence_regions': len(bbox_evidence['primary_regions']),
                'coverage_ratio': self._calculate_bbox_coverage_ratio(sorted_regions, image_size),
                'concentration_index': self._calculate_bbox_concentration_index(sorted_regions),
                'spatial_spread': self._calculate_bbox_spatial_spread(sorted_regions, image_size)
            }
            
            # Localization precision metrics
            all_processed_regions = bbox_evidence['primary_regions'] + bbox_evidence['secondary_regions']
            if all_processed_regions:
                bbox_evidence['localization_precision'] = {
                    'average_region_size': np.mean([r['relative_size'] for r in all_processed_regions[:5]]),
                    'size_variance': np.var([r['relative_size'] for r in all_processed_regions[:5]]) if len(all_processed_regions) > 1 else 0.0,
                    'precision_score': self._calculate_localization_precision(sorted_regions)
                }
            else:
                bbox_evidence['localization_precision'] = {
                    'average_region_size': 0.0,
                    'size_variance': 0.0,
                    'precision_score': 0.0
                }
            
            # Region characteristics
            primary_region = sorted_regions[0]
            bbox_evidence['region_characteristics'] = {
                'dominant_region': {
                    'score': primary_region.get('attention_score', primary_region.get('score', 0)),
                    'location': self._describe_spatial_location(
                        self._calculate_region_center(primary_region['bbox']), image_size
                    ),
                    'size_category': self._categorize_bbox_size(primary_region['bbox'], image_size)
                },
                'region_diversity': len(set(self._describe_spatial_location(
                    self._calculate_region_center(r['bbox']), image_size
                ) for r in sorted_regions[:3]))
            }
        
        return bbox_evidence
    
    def _extract_attention_evidence(self, attention_regions: List[Dict], 
                                   image_size: Tuple[int, int]) -> Dict:
        """PRESERVED: Extract evidence from basic attention regions (fallback)"""
        attention_evidence = {
            'primary_regions': [],
            'secondary_regions': [],
            'attention_distribution': {},
            'spatial_focus': {}
        }
        
        # Sort regions by attention score
        sorted_regions = sorted(attention_regions, key=lambda x: x.get('score', 0), reverse=True)
        
        # Categorize regions by attention strength
        for region in sorted_regions:
            score = region.get('score', 0)
            region_info = {
                'bbox': region.get('bbox', [0, 0, 0, 0]),
                'center': self._calculate_region_center(region.get('bbox', [0, 0, 0, 0])),
                'score': score,
                'relative_size': self._calculate_relative_size(region, image_size),
                'strength': self._categorize_attention_strength(score)
            }
            
            if score >= self.attention_thresholds['strong']:
                attention_evidence['primary_regions'].append(region_info)
            elif score >= self.attention_thresholds['moderate']:
                attention_evidence['secondary_regions'].append(region_info)
        
        # Calculate attention distribution
        total_score = sum(r.get('score', 0) for r in sorted_regions)
        if total_score > 0:
            attention_evidence['attention_distribution'] = {
                'concentration_index': self._calculate_concentration_index(sorted_regions),
                'spatial_spread': self._calculate_spatial_spread(sorted_regions, image_size),
                'focus_intensity': sorted_regions[0].get('score', 0) if sorted_regions else 0
            }
        
        # Determine spatial focus characteristics
        if attention_evidence['primary_regions']:
            primary_region = attention_evidence['primary_regions'][0]
            attention_evidence['spatial_focus'] = {
                'location': self._describe_spatial_location(primary_region['center'], image_size),
                'extent': self._describe_region_extent(primary_region),
                'confidence': primary_region['score']
            }
        
        return attention_evidence
    
    def _extract_spatial_evidence(self, spatial_patterns: Dict, 
                                 image_size: Tuple[int, int]) -> Dict:
        """PRESERVED: Extract evidence from spatial patterns"""
        spatial_evidence = {
            'attention_map_analysis': {},
            'focus_regions_analysis': {},
            'spatial_relationships': {}
        }
        
        # Analyze attention map if available
        if 'attention_map' in spatial_patterns:
            attention_map = spatial_patterns['attention_map']
            spatial_evidence['attention_map_analysis'] = {
                'entropy': spatial_patterns.get('attention_entropy', 0),
                'peak_locations': self._find_attention_peaks(attention_map),
                'distribution_type': self._classify_attention_distribution(
                    spatial_patterns.get('attention_entropy', 0)
                )
            }
        
        # Analyze focus regions
        if 'focus_regions' in spatial_patterns:
            focus_regions = spatial_patterns['focus_regions']
            spatial_evidence['focus_regions_analysis'] = {
                'region_count': len(focus_regions),
                'primary_focus': focus_regions[0] if focus_regions else None,
                'secondary_foci': focus_regions[1:] if len(focus_regions) > 1 else [],
                'spatial_clustering': self._analyze_spatial_clustering(focus_regions)
            }
        
        return spatial_evidence
    
    def _extract_feature_evidence(self, visual_description: str, 
                                 anatomical_context: str) -> Dict:
        """PRESERVED: Extract evidence from feature descriptions"""
        feature_evidence = {
            'visual_descriptors': [],
            'anatomical_indicators': [],
            'pathological_features': [],
            'confidence_indicators': {}
        }
        
        # Parse visual description for evidence
        description_lower = visual_description.lower()
        
        # Extract visual descriptors
        visual_keywords = [
            'complexity', 'attention', 'focus', 'regions', 'distributed',
            'concentrated', 'pattern', 'structure', 'appearance'
        ]
        
        for keyword in visual_keywords:
            if keyword in description_lower:
                feature_evidence['visual_descriptors'].append(keyword)
        
        # Extract anatomical indicators
        anatomical_keywords = [
            'anatomical', 'tissue', 'organ', 'structure', 'region',
            'location', 'system', 'anatomy'
        ]
        
        for keyword in anatomical_keywords:
            if keyword in anatomical_context.lower():
                feature_evidence['anatomical_indicators'].append(keyword)
        
        # Extract pathological features
        pathology_keywords = [
            'pathology', 'abnormal', 'lesion', 'mass', 'inflammation',
            'necrosis', 'ischemia', 'tumor', 'infection'
        ]
        
        for keyword in pathology_keywords:
            if keyword in description_lower or keyword in anatomical_context.lower():
                feature_evidence['pathological_features'].append(keyword)
        
        # Assess confidence indicators
        feature_evidence['confidence_indicators'] = {
            'visual_complexity': 'high' if 'complexity' in description_lower else 'moderate',
            'anatomical_specificity': 'high' if len(feature_evidence['anatomical_indicators']) > 2 else 'moderate',
            'pathological_evidence': 'high' if len(feature_evidence['pathological_features']) > 1 else 'moderate'
        }
        
        return feature_evidence
    
    def link_evidence_to_reasoning_step(self, reasoning_step: Dict, 
                                      visual_evidence: Dict) -> Dict:
        """
        🆕 ENHANCED: Link visual evidence including bounding boxes to reasoning steps
        
        Args:
            reasoning_step: Dictionary containing reasoning step information
            visual_evidence: Enhanced visual evidence dictionary with bounding box support
            
        Returns:
            Reasoning step with enhanced evidence links
        """
        step_type = reasoning_step.get('type', 'unknown')
        enhanced_step = reasoning_step.copy()
        
        # Initialize enhanced evidence links
        enhanced_step['evidence_links'] = {
            'visual_support': [],
            'attention_support': [],
            'spatial_support': [],
            # 🆕 NEW: Bounding box evidence links
            'bounding_box_support': [],
            'spatial_localization': [],
            'confidence_modifiers': []
        }
        
        # 🆕 ENHANCED: Link evidence based on step type with bounding box priority
        if step_type == 'visual_observation':
            enhanced_step['evidence_links']['visual_support'] = self._link_visual_observation_evidence(
                reasoning_step, visual_evidence
            )
            # 🆕 NEW: Add bounding box support for visual observations
            enhanced_step['evidence_links']['bounding_box_support'] = self._link_bounding_box_evidence(
                reasoning_step, visual_evidence
            )
        
        elif step_type == 'attention_analysis':
            enhanced_step['evidence_links']['attention_support'] = self._link_attention_evidence(
                reasoning_step, visual_evidence
            )
            # 🆕 NEW: Enhanced with bounding box spatial localization
            enhanced_step['evidence_links']['spatial_localization'] = self._link_spatial_localization_evidence(
                reasoning_step, visual_evidence
            )
        
        elif step_type == 'spatial_analysis':
            enhanced_step['evidence_links']['spatial_support'] = self._link_spatial_evidence(
                reasoning_step, visual_evidence
            )
            enhanced_step['evidence_links']['bounding_box_support'] = self._link_bounding_box_evidence(
                reasoning_step, visual_evidence
            )
        
        elif step_type in ['clinical_correlation', 'diagnostic_reasoning']:
            enhanced_step['evidence_links']['visual_support'] = self._link_clinical_evidence(
                reasoning_step, visual_evidence
            )
            # 🆕 NEW: Add spatial evidence for clinical correlation
            enhanced_step['evidence_links']['spatial_localization'] = self._link_spatial_localization_evidence(
                reasoning_step, visual_evidence
            )
        
        # 🆕 ENHANCED: Calculate confidence modifiers with bounding box consideration
        enhanced_step['evidence_links']['confidence_modifiers'] = self._calculate_enhanced_evidence_confidence(
            enhanced_step['evidence_links'], visual_evidence
        )
        
        # Update step confidence based on enhanced evidence
        original_confidence = reasoning_step.get('confidence', 0.5)
        evidence_confidence = enhanced_step['evidence_links']['confidence_modifiers'].get('overall', 1.0)
        enhanced_step['confidence'] = min(original_confidence * evidence_confidence, 1.0)
        
        return enhanced_step
    
    def _link_bounding_box_evidence(self, reasoning_step: Dict, 
                                   visual_evidence: Dict) -> List[Dict]:
        """
        🆕 NEW: Link bounding box evidence to reasoning steps
        """
        evidence_links = []
        
        # Link bounding box evidence if available
        if 'bounding_box_evidence' in visual_evidence:
            bbox_data = visual_evidence['bounding_box_evidence']
            
            # Primary bounding box regions
            if bbox_data.get('primary_regions'):
                evidence_links.append({
                    'type': 'primary_bounding_boxes',
                    'data': bbox_data['primary_regions'],
                    'relevance': 'very_high',
                    'description': f"High-confidence spatial attention regions with precise localization ({len(bbox_data['primary_regions'])} primary regions)"
                })
            
            # Spatial distribution analysis
            if bbox_data.get('spatial_distribution'):
                evidence_links.append({
                    'type': 'spatial_distribution_analysis',
                    'data': bbox_data['spatial_distribution'],
                    'relevance': 'high',
                    'description': 'Comprehensive spatial distribution analysis of attention regions'
                })
            
            # Localization precision
            if bbox_data.get('localization_precision'):
                evidence_links.append({
                    'type': 'localization_precision',
                    'data': bbox_data['localization_precision'],
                    'relevance': 'high',
                    'description': 'Quantitative precision metrics for spatial localization'
                })
        
        return evidence_links
    
    def _link_spatial_localization_evidence(self, reasoning_step: Dict, 
                                          visual_evidence: Dict) -> List[Dict]:
        """
        🆕 NEW: Link spatial localization evidence specifically for reasoning steps
        """
        evidence_links = []
        
        # Check for bounding box evidence first (highest priority)
        if 'bounding_box_evidence' in visual_evidence:
            bbox_data = visual_evidence['bounding_box_evidence']
            
            if bbox_data.get('region_characteristics'):
                evidence_links.append({
                    'type': 'dominant_region_characteristics',
                    'data': bbox_data['region_characteristics'],
                    'relevance': 'very_high',
                    'description': 'Characteristics of the dominant attention region with precise spatial localization'
                })
        
        # Fallback to basic attention evidence
        elif 'attention_evidence' in visual_evidence:
            attention_data = visual_evidence['attention_evidence']
            
            if attention_data.get('spatial_focus'):
                evidence_links.append({
                    'type': 'spatial_focus_basic',
                    'data': attention_data['spatial_focus'],
                    'relevance': 'high',
                    'description': 'Basic spatial focus characteristics from attention analysis'
                })
        
        return evidence_links
    
    def _link_visual_observation_evidence(self, reasoning_step: Dict, 
                                        visual_evidence: Dict) -> List[Dict]:
        """PRESERVED: Link evidence for visual observation steps"""
        evidence_links = []
        
        # Link image metadata
        if 'image_metadata' in visual_evidence:
            evidence_links.append({
                'type': 'image_characteristics',
                'data': visual_evidence['image_metadata'],
                'relevance': 'high',
                'description': 'Basic image characteristics supporting observation'
            })
        
        # Link feature evidence
        if 'feature_evidence' in visual_evidence:
            feature_data = visual_evidence['feature_evidence']
            if feature_data.get('visual_descriptors'):
                evidence_links.append({
                    'type': 'visual_features',
                    'data': feature_data['visual_descriptors'],
                    'relevance': 'high',
                    'description': 'Visual features identified in the image'
                })
        
        return evidence_links
    
    def _link_attention_evidence(self, reasoning_step: Dict, 
                               visual_evidence: Dict) -> List[Dict]:
        """ENHANCED: Link attention evidence with bounding box priority"""
        evidence_links = []
        
        # Priority 1: Bounding box evidence
        if 'bounding_box_evidence' in visual_evidence:
            bbox_data = visual_evidence['bounding_box_evidence']
            
            if bbox_data.get('primary_regions'):
                evidence_links.append({
                    'type': 'primary_bbox_attention',
                    'data': bbox_data['primary_regions'],
                    'relevance': 'very_high',
                    'description': 'Primary regions of precise spatial attention with bounding boxes'
                })
        
        # Priority 2: Basic attention evidence (fallback)
        elif 'attention_evidence' in visual_evidence:
            attention_data = visual_evidence['attention_evidence']
            
            if attention_data.get('primary_regions'):
                evidence_links.append({
                    'type': 'primary_attention_basic',
                    'data': attention_data['primary_regions'],
                    'relevance': 'high',
                    'description': 'Primary regions of model attention (basic analysis)'
                })
        
        return evidence_links
    
    def _link_spatial_evidence(self, reasoning_step: Dict, 
                             visual_evidence: Dict) -> List[Dict]:
        """PRESERVED: Link evidence for spatial analysis steps"""
        evidence_links = []
        
        # Link spatial evidence
        if 'spatial_evidence' in visual_evidence:
            spatial_data = visual_evidence['spatial_evidence']
            
            # Attention map analysis
            if spatial_data.get('attention_map_analysis'):
                evidence_links.append({
                    'type': 'attention_distribution',
                    'data': spatial_data['attention_map_analysis'],
                    'relevance': 'high',
                    'description': 'Spatial distribution analysis of attention'
                })
            
            # Focus regions analysis
            if spatial_data.get('focus_regions_analysis'):
                evidence_links.append({
                    'type': 'focus_analysis',
                    'data': spatial_data['focus_regions_analysis'],
                    'relevance': 'high',
                    'description': 'Analysis of attention focus regions'
                })
        
        return evidence_links
    
    def _link_clinical_evidence(self, reasoning_step: Dict, 
                              visual_evidence: Dict) -> List[Dict]:
        """PRESERVED: Link evidence for clinical correlation steps"""
        evidence_links = []
        
        # Link pathological features
        if 'feature_evidence' in visual_evidence:
            feature_data = visual_evidence['feature_evidence']
            
            if feature_data.get('pathological_features'):
                evidence_links.append({
                    'type': 'pathological_indicators',
                    'data': feature_data['pathological_features'],
                    'relevance': 'high',
                    'description': 'Pathological features identified in the analysis'
                })
            
            if feature_data.get('anatomical_indicators'):
                evidence_links.append({
                    'type': 'anatomical_context',
                    'data': feature_data['anatomical_indicators'],
                    'relevance': 'moderate',
                    'description': 'Anatomical context supporting clinical correlation'
                })
        
        return evidence_links
    
    def _calculate_enhanced_evidence_confidence(self, evidence_links: Dict, 
                                              visual_evidence: Dict) -> Dict:
        """
        🆕 ENHANCED: Calculate confidence modifiers with bounding box consideration
        """
        confidence_modifiers = {
            'visual_support_strength': 1.0,
            'attention_support_strength': 1.0,
            'spatial_support_strength': 1.0,
            # 🆕 NEW: Bounding box confidence modifiers
            'bounding_box_support_strength': 1.0,
            'spatial_localization_strength': 1.0,
            'overall': 1.0
        }
        
        # Calculate visual support strength
        visual_support = evidence_links.get('visual_support', [])
        if visual_support:
            high_relevance_count = sum(1 for link in visual_support if link.get('relevance') == 'high')
            confidence_modifiers['visual_support_strength'] = min(1.0, high_relevance_count * 0.3 + 0.4)
        
        # 🆕 ENHANCED: Calculate bounding box support strength (highest weight)
        bbox_support = evidence_links.get('bounding_box_support', [])
        if bbox_support:
            bbox_strength = 0.6  # Higher base for bounding boxes
            for link in bbox_support:
                if link.get('type') == 'primary_bounding_boxes':
                    primary_regions = link.get('data', [])
                    if primary_regions:
                        # Use bounding box attention scores
                        max_score = max(region.get('attention_score', region.get('score', 0)) for region in primary_regions)
                        bbox_strength += max_score * 0.3  # Higher multiplier for bbox
                elif link.get('relevance') == 'very_high':
                    bbox_strength += 0.2
            confidence_modifiers['bounding_box_support_strength'] = min(bbox_strength, 1.0)
        
        # Calculate attention support strength (consider both bbox and basic)
        attention_support = evidence_links.get('attention_support', [])
        if attention_support:
            attention_strength = 0.5
            for link in attention_support:
                if link.get('type') == 'primary_bbox_attention':
                    # Higher weight for bbox attention
                    primary_regions = link.get('data', [])
                    if primary_regions:
                        max_score = max(region.get('attention_score', region.get('score', 0)) for region in primary_regions)
                        attention_strength += max_score * 0.4
                elif link.get('type') == 'primary_attention_basic':
                    # Standard weight for basic attention
                    primary_regions = link.get('data', [])
                    if primary_regions:
                        max_score = max(region.get('score', 0) for region in primary_regions)
                        attention_strength += max_score * 0.3
            confidence_modifiers['attention_support_strength'] = min(attention_strength, 1.0)
        
        # 🆕 NEW: Calculate spatial localization strength
        spatial_localization = evidence_links.get('spatial_localization', [])
        if spatial_localization:
            localization_strength = 0.5
            for link in spatial_localization:
                if link.get('relevance') == 'very_high':
                    localization_strength += 0.3
                elif link.get('relevance') == 'high':
                    localization_strength += 0.2
            confidence_modifiers['spatial_localization_strength'] = min(localization_strength, 1.0)
        
        # Calculate spatial support strength
        spatial_support = evidence_links.get('spatial_support', [])
        if spatial_support:
            spatial_strength = 0.5
            for link in spatial_support:
                if link.get('relevance') == 'high':
                    spatial_strength += 0.25
            confidence_modifiers['spatial_support_strength'] = min(spatial_strength, 1.0)
        
        # 🆕 ENHANCED: Calculate overall confidence with bounding box priority
        individual_confidences = [
            confidence_modifiers['visual_support_strength'],
            confidence_modifiers['attention_support_strength'],
            confidence_modifiers['spatial_support_strength']
        ]
        
        # Add bounding box confidences with higher weight if available
        if confidence_modifiers['bounding_box_support_strength'] > 0.6:
            individual_confidences.extend([
                confidence_modifiers['bounding_box_support_strength'] * 1.2,  # Higher weight
                confidence_modifiers['spatial_localization_strength']
            ])
        
        confidence_modifiers['overall'] = sum(individual_confidences) / len(individual_confidences)
        
        return confidence_modifiers
    
    # 🆕 NEW: Bounding box utility methods
    def _calculate_bbox_relative_size(self, bbox: List[int], image_size: Tuple[int, int]) -> float:
        """Calculate relative size of bounding box compared to image"""
        if len(bbox) >= 4:
            _, _, w, h = bbox[:4]
            bbox_area = w * h
            image_area = image_size[0] * image_size[1]
            return bbox_area / image_area if image_area > 0 else 0
        return 0
    
    def _categorize_bbox_attention_strength(self, score: float) -> str:
        """Categorize bounding box attention strength"""
        if score >= self.bbox_thresholds['high_confidence']:
            return 'very_strong'
        elif score >= self.bbox_thresholds['medium_confidence']:
            return 'strong'
        elif score >= self.bbox_thresholds['low_confidence']:
            return 'moderate'
        else:
            return 'weak'
    
    def _describe_bbox_extent(self, bbox: List[int], image_size: Tuple[int, int]) -> str:
        """Describe the extent/size of a bounding box"""
        relative_size = self._calculate_bbox_relative_size(bbox, image_size)
        
        if relative_size > 0.25:
            return "large"
        elif relative_size > 0.1:
            return "moderate"
        elif relative_size > 0.05:
            return "small"
        else:
            return "focal"
    
    def _categorize_bbox_size(self, bbox: List[int], image_size: Tuple[int, int]) -> str:
        """Categorize bounding box size"""
        relative_size = self._calculate_bbox_relative_size(bbox, image_size)
        
        if relative_size > 0.3:
            return "extensive"
        elif relative_size > 0.15:
            return "substantial"
        elif relative_size > 0.05:
            return "moderate"
        else:
            return "localized"
    
    def _calculate_bbox_coverage_ratio(self, bbox_regions: List[Dict], 
                                     image_size: Tuple[int, int]) -> float:
        """Calculate total coverage ratio of all bounding boxes"""
        total_area = 0
        image_area = image_size[0] * image_size[1]
        
        for region in bbox_regions:
            bbox = region.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                _, _, w, h = bbox[:4]
                total_area += w * h
        
        return min(total_area / image_area, 1.0) if image_area > 0 else 0
    
    def _calculate_bbox_concentration_index(self, bbox_regions: List[Dict]) -> float:
        """Calculate concentration index for bounding boxes"""
        if not bbox_regions:
            return 0
        
        scores = [r.get('attention_score', r.get('score', 0)) for r in bbox_regions]
        total_score = sum(scores)
        
        if total_score == 0:
            return 0
        
        # Calculate entropy-based concentration
        normalized_scores = [s/total_score for s in scores]
        entropy = -sum(p * np.log(p + 1e-8) for p in normalized_scores if p > 0)
        max_entropy = np.log(len(scores))
        
        return 1 - (entropy / max_entropy) if max_entropy > 0 else 0
    
    def _calculate_bbox_spatial_spread(self, bbox_regions: List[Dict], 
                                     image_size: Tuple[int, int]) -> float:
        """Calculate spatial spread of bounding box regions"""
        if len(bbox_regions) < 2:
            return 0
        
        centers = [self._calculate_region_center(r.get('bbox', [0, 0, 0, 0])) for r in bbox_regions]
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                             (centers[i][1] - centers[j][1])**2)
                distances.append(dist)
        
        # Normalize by image diagonal
        max_distance = np.sqrt(image_size[0]**2 + image_size[1]**2)
        avg_distance = np.mean(distances) if distances else 0
        
        return avg_distance / max_distance if max_distance > 0 else 0
    
    def _calculate_localization_precision(self, bbox_regions: List[Dict]) -> float:
        """Calculate overall localization precision score"""
        if not bbox_regions:
            return 0
        
        # Factors: attention score distribution, size consistency, spatial organization
        scores = [r.get('attention_score', r.get('score', 0)) for r in bbox_regions]
        sizes = [r.get('relative_size', 0) for r in bbox_regions if 'relative_size' in r]
        
        # Score consistency (higher is better)
        score_consistency = 1 - np.std(scores) if len(scores) > 1 else 1.0
        
        # Size appropriateness (moderate sizes are better for precision)
        if sizes:
            size_appropriateness = 1 - abs(np.mean(sizes) - 0.1)  # Target ~10% of image
        else:
            size_appropriateness = 0.5
        
        # Combine factors
        precision_score = (score_consistency * 0.6 + size_appropriateness * 0.4)
        return max(0, min(precision_score, 1.0))
    
    # PRESERVED: Existing utility methods
    def _calculate_region_center(self, bbox: List[int]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        if len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            return (x + w/2, y + h/2)
        return (0, 0)
    
    def _calculate_relative_size(self, region: Dict, image_size: Tuple[int, int]) -> float:
        """Calculate relative size of region compared to image"""
        bbox = region.get('bbox', [0, 0, 0, 0])
        if len(bbox) >= 4:
            _, _, w, h = bbox[:4]
            region_area = w * h
            image_area = image_size[0] * image_size[1]
            return region_area / image_area if image_area > 0 else 0
        return 0
    
    def _categorize_attention_strength(self, score: float) -> str:
        """Categorize attention strength based on score"""
        if score >= self.attention_thresholds['strong']:
            return 'strong'
        elif score >= self.attention_thresholds['moderate']:
            return 'moderate'
        elif score >= self.attention_thresholds['weak']:
            return 'weak'
        else:
            return 'minimal'
    
    def _calculate_concentration_index(self, regions: List[Dict]) -> float:
        """Calculate how concentrated the attention is"""
        if not regions:
            return 0
        
        scores = [r.get('score', 0) for r in regions]
        total_score = sum(scores)
        
        if total_score == 0:
            return 0
        
        # Calculate entropy-based concentration
        normalized_scores = [s/total_score for s in scores]
        entropy = -sum(p * np.log(p + 1e-8) for p in normalized_scores if p > 0)
        max_entropy = np.log(len(scores))
        
        # Convert to concentration (inverse of normalized entropy)
        return 1 - (entropy / max_entropy) if max_entropy > 0 else 0
    
    def _calculate_spatial_spread(self, regions: List[Dict], 
                                image_size: Tuple[int, int]) -> float:
        """Calculate spatial spread of attention regions"""
        if len(regions) < 2:
            return 0
        
        centers = [self._calculate_region_center(r.get('bbox', [0, 0, 0, 0])) for r in regions]
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                             (centers[i][1] - centers[j][1])**2)
                distances.append(dist)
        
        # Normalize by image diagonal
        max_distance = np.sqrt(image_size[0]**2 + image_size[1]**2)
        avg_distance = np.mean(distances) if distances else 0
        
        return avg_distance / max_distance if max_distance > 0 else 0
    
    def _describe_spatial_location(self, center: Tuple[float, float], 
                                 image_size: Tuple[int, int]) -> str:
        """Describe spatial location in human-readable terms"""
        x, y = center
        width, height = image_size
        
        # Determine horizontal position
        if x < width * 0.33:
            h_pos = "left"
        elif x > width * 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Determine vertical position
        if y < height * 0.33:
            v_pos = "upper"
        elif y > height * 0.67:
            v_pos = "lower"
        else:
            v_pos = "middle"
        
        return f"{v_pos} {h_pos}"
    
    def _describe_region_extent(self, region_info: Dict) -> str:
        """Describe the extent/size of a region"""
        relative_size = region_info.get('relative_size', 0)
        
        if relative_size > 0.3:
            return "large"
        elif relative_size > 0.1:
            return "moderate"
        elif relative_size > 0.05:
            return "small"
        else:
            return "focal"
    
    def _find_attention_peaks(self, attention_map: np.ndarray) -> List[Tuple[int, int]]:
        """Find peak locations in attention map"""
        try:
            from scipy import ndimage
            
            # Find local maxima
            local_maxima = ndimage.maximum_filter(attention_map, size=3) == attention_map
            peaks = np.where(local_maxima & (attention_map > np.percentile(attention_map, 90)))
            
            return list(zip(peaks[1], peaks[0]))  # (x, y) coordinates
        except ImportError:
            return []
    
    def _classify_attention_distribution(self, entropy: float) -> str:
        """Classify attention distribution type based on entropy"""
        if entropy > 2.5:
            return "distributed"
        elif entropy > 1.5:
            return "moderate"
        else:
            return "focused"
    
    def _analyze_spatial_clustering(self, focus_regions: List[Dict]) -> Dict:
        """Analyze spatial clustering of focus regions"""
        if len(focus_regions) < 2:
            return {'type': 'single', 'clusters': 1}
        
        # Simple clustering analysis based on region centers
        centers = [r.get('center', [0, 0]) for r in focus_regions]
        
        # Calculate average distance between regions
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                             (centers[i][1] - centers[j][1])**2)
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # Simple clustering classification
        if avg_distance < 50:  # Close together
            return {'type': 'clustered', 'clusters': 1, 'avg_distance': avg_distance}
        elif avg_distance < 100:  # Moderate separation
            return {'type': 'moderate', 'clusters': 2, 'avg_distance': avg_distance}
        else:  # Widely separated
            return {'type': 'distributed', 'clusters': len(focus_regions), 'avg_distance': avg_distance}
    
    def _create_evidence_summary(self, evidence: Dict) -> Dict:
        """🆕 ENHANCED: Create summary with bounding box priority"""
        summary = {
            'total_evidence_sources': 0,
            'primary_evidence_types': [],
            'confidence_level': 'moderate',
            'key_findings': [],
            # 🆕 NEW: Bounding box summary
            'has_bounding_boxes': False,
            'spatial_precision': 'unknown'
        }
        
        # Count evidence sources with bounding box priority
        evidence_types_to_check = ['bounding_box_evidence', 'attention_evidence', 'spatial_evidence', 'feature_evidence']
        
        for evidence_type in evidence_types_to_check:
            if evidence_type in evidence and evidence[evidence_type]:
                summary['total_evidence_sources'] += 1
                summary['primary_evidence_types'].append(evidence_type)
        
        # 🆕 ENHANCED: Determine confidence level with bounding box boost
        if 'bounding_box_evidence' in evidence and evidence['bounding_box_evidence']:
            summary['has_bounding_boxes'] = True
            summary['confidence_level'] = 'high'  # Bounding boxes boost confidence
            
            # Assess spatial precision
            bbox_data = evidence['bounding_box_evidence']
            if bbox_data.get('localization_precision', {}).get('precision_score', 0) > 0.7:
                summary['spatial_precision'] = 'high'
            elif bbox_data.get('localization_precision', {}).get('precision_score', 0) > 0.5:
                summary['spatial_precision'] = 'moderate'
            else:
                summary['spatial_precision'] = 'low'
        elif summary['total_evidence_sources'] >= 3:
            summary['confidence_level'] = 'high'
        elif summary['total_evidence_sources'] >= 2:
            summary['confidence_level'] = 'moderate'
        else:
            summary['confidence_level'] = 'low'
        
        # Extract key findings with bounding box priority
        if 'bounding_box_evidence' in evidence:
            bbox_data = evidence['bounding_box_evidence']
            if bbox_data.get('primary_regions'):
                primary_count = len(bbox_data['primary_regions'])
                total_count = bbox_data.get('spatial_distribution', {}).get('total_regions', 0)
                summary['key_findings'].append(f"High-precision spatial attention detected: {primary_count} high-confidence regions out of {total_count} total bounding boxes")
        elif 'attention_evidence' in evidence:
            attention_data = evidence['attention_evidence']
            if attention_data.get('primary_regions'):
                summary['key_findings'].append(f"Basic attention focus detected in {len(attention_data['primary_regions'])} primary regions")
        
        if 'feature_evidence' in evidence:
            feature_data = evidence['feature_evidence']
            if feature_data.get('pathological_features'):
                summary['key_findings'].append(f"Pathological features identified: {', '.join(feature_data['pathological_features'])}")
        
        return summary
