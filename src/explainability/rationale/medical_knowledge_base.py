import logging
from typing import Dict, List, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)

class MedicalKnowledgeBase:
    """
    Medical Knowledge Base for Chain-of-Thought Reasoning
    Maps visual features to clinical terminology and diagnostic patterns
    """
    
    def __init__(self, config):
        """
        Initialize Medical Knowledge Base
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize knowledge bases
        self.anatomical_structures = self._init_anatomical_structures()
        self.pathology_patterns = self._init_pathology_patterns()
        self.diagnostic_criteria = self._init_diagnostic_criteria()
        self.visual_to_clinical_mapping = self._init_visual_clinical_mapping()
        self.reasoning_patterns = self._init_reasoning_patterns()
        
        logger.info("Medical Knowledge Base initialized")
    
    def _init_anatomical_structures(self) -> Dict:
        """Initialize anatomical structure knowledge"""
        return {
            'cardiovascular': {
                'organs': ['heart', 'blood vessels', 'arteries', 'veins'],
                'tissues': ['myocardium', 'endocardium', 'pericardium'],
                'cells': ['cardiomyocytes', 'endothelial cells'],
                'visual_indicators': ['cardiac silhouette', 'vessel caliber', 'chamber size']
            },
            'respiratory': {
                'organs': ['lungs', 'bronchi', 'trachea'],
                'tissues': ['alveolar tissue', 'bronchial tissue', 'pleura'],
                'cells': ['pneumocytes', 'alveolar macrophages'],
                'visual_indicators': ['lung fields', 'bronchial markings', 'pleural line']
            },
            'hepatic': {
                'organs': ['liver', 'gallbladder', 'bile ducts'],
                'tissues': ['hepatic parenchyma', 'portal tracts'],
                'cells': ['hepatocytes', 'kupffer cells', 'stellate cells'],
                'visual_indicators': ['liver echotexture', 'portal vasculature', 'bile duct dilation']
            },
            'renal': {
                'organs': ['kidneys', 'ureters', 'bladder'],
                'tissues': ['cortex', 'medulla', 'glomeruli'],
                'cells': ['nephrons', 'tubular cells'],
                'visual_indicators': ['cortical thickness', 'medullary pyramids', 'collecting system']
            },
            'musculoskeletal': {
                'organs': ['bones', 'joints', 'muscles'],
                'tissues': ['cortical bone', 'trabecular bone', 'cartilage'],
                'cells': ['osteocytes', 'chondrocytes', 'myocytes'],
                'visual_indicators': ['bone density', 'joint space', 'soft tissue swelling']
            },
            'nervous': {
                'organs': ['brain', 'spinal cord', 'nerves'],
                'tissues': ['gray matter', 'white matter', 'meninges'],
                'cells': ['neurons', 'glial cells'],
                'visual_indicators': ['brain atrophy', 'ventricular size', 'lesion characteristics']
            }
        }
    
    def _init_pathology_patterns(self) -> Dict:
        """Initialize pathology pattern knowledge"""
        return {
            'inflammation': {
                'acute': {
                    'visual_features': ['erythema', 'swelling', 'increased vascularity'],
                    'cellular_changes': ['neutrophil infiltration', 'edema', 'hyperemia'],
                    'typical_locations': ['infection sites', 'trauma areas', 'autoimmune targets']
                },
                'chronic': {
                    'visual_features': ['fibrosis', 'tissue remodeling', 'architectural distortion'],
                    'cellular_changes': ['lymphocyte infiltration', 'macrophage accumulation', 'fibroblast proliferation'],
                    'typical_locations': ['persistent irritation sites', 'autoimmune organs']
                }
            },
            'neoplasia': {
                'benign': {
                    'visual_features': ['well-demarcated borders', 'homogeneous appearance', 'slow growth'],
                    'cellular_changes': ['uniform cell morphology', 'low mitotic activity'],
                    'typical_locations': ['organ-specific sites', 'encapsulated masses']
                },
                'malignant': {
                    'visual_features': ['irregular borders', 'heterogeneous appearance', 'rapid growth'],
                    'cellular_changes': ['pleomorphic cells', 'high mitotic activity', 'invasion'],
                    'typical_locations': ['primary sites', 'metastatic locations']
                }
            },
            'ischemia': {
                'acute': {
                    'visual_features': ['tissue pallor', 'loss of normal architecture'],
                    'cellular_changes': ['cell swelling', 'nuclear pyknosis', 'cytoplasmic eosinophilia'],
                    'typical_locations': ['vascular territories', 'watershed areas']
                },
                'chronic': {
                    'visual_features': ['atrophy', 'fibrosis', 'collateral circulation'],
                    'cellular_changes': ['cell loss', 'replacement fibrosis'],
                    'typical_locations': ['end-organ territories']
                }
            },
            'degeneration': {
                'fatty_change': {
                    'visual_features': ['increased echogenicity', 'tissue brightening'],
                    'cellular_changes': ['lipid accumulation', 'hepatocyte ballooning'],
                    'typical_locations': ['liver', 'heart', 'kidney']
                },
                'necrosis': {
                    'visual_features': ['tissue darkening', 'loss of enhancement'],
                    'cellular_changes': ['cell death', 'nuclear fragmentation'],
                    'typical_locations': ['ischemic zones', 'toxic injury sites']
                }
            }
        }
    
    def _init_diagnostic_criteria(self) -> Dict:
        """Initialize diagnostic criteria knowledge"""
        return {
            'early_ischemic_injury': {
                'primary_criteria': [
                    'increased cytoplasmic eosinophilia',
                    'cellular swelling',
                    'nuclear changes (pyknosis, karyorrhexis)'
                ],
                'secondary_criteria': [
                    'loss of cellular detail',
                    'tissue architecture preservation (early stage)',
                    'minimal inflammatory response'
                ],
                'differential_diagnosis': [
                    'acute inflammation',
                    'toxic injury',
                    'heat shock response'
                ],
                'reversibility': 'potentially reversible if reperfusion occurs early'
            },
            'hepatocellular_carcinoma': {
                'primary_criteria': [
                    'arterial hypervascularity',
                    'portal/delayed phase washout',
                    'capsule appearance'
                ],
                'secondary_criteria': [
                    'threshold growth',
                    'corona enhancement',
                    'mosaic architecture'
                ],
                'differential_diagnosis': [
                    'metastatic disease',
                    'cholangiocarcinoma',
                    'benign liver lesions'
                ]
            },
            'acute_myocardial_infarction': {
                'primary_criteria': [
                    'regional wall motion abnormality',
                    'myocardial edema (T2 hyperintensity)',
                    'late gadolinium enhancement'
                ],
                'secondary_criteria': [
                    'microvascular obstruction',
                    'hemorrhage',
                    'pericardial effusion'
                ],
                'differential_diagnosis': [
                    'myocarditis',
                    'takotsubo cardiomyopathy',
                    'cardiac contusion'
                ]
            }
        }
    
    def _init_visual_clinical_mapping(self) -> Dict:
        """Initialize visual feature to clinical term mapping"""
        return {
            'density_changes': {
                'hypodense': ['necrosis', 'edema', 'cystic change', 'fat'],
                'isodense': ['normal tissue', 'isoattenuating lesion'],
                'hyperdense': ['hemorrhage', 'calcification', 'contrast enhancement', 'fibrosis']
            },
            'enhancement_patterns': {
                'arterial_enhancement': ['hypervascular lesions', 'inflammation', 'malignancy'],
                'portal_enhancement': ['normal parenchyma', 'some benign lesions'],
                'delayed_enhancement': ['fibrosis', 'some malignancies'],
                'no_enhancement': ['necrosis', 'cysts', 'avascular lesions']
            },
            'morphological_features': {
                'well_demarcated': ['benign lesions', 'abscesses', 'cysts'],
                'ill_defined': ['malignancy', 'inflammation', 'infiltrative process'],
                'lobulated': ['benign masses', 'some malignancies'],
                'spiculated': ['malignancy', 'sclerosing processes']
            },
            'signal_characteristics': {
                't1_hyperintense': ['fat', 'hemorrhage', 'protein-rich fluid', 'melanin'],
                't1_hypointense': ['simple fluid', 'edema', 'most pathology'],
                't2_hyperintense': ['fluid', 'edema', 'inflammation', 'most pathology'],
                't2_hypointense': ['fibrosis', 'calcification', 'hemosiderin', 'air']
            }
        }
    
    def _init_reasoning_patterns(self) -> Dict:
        """Initialize clinical reasoning patterns"""
        return {
            'observation_to_finding': {
                'pattern': 'Visual observation of {visual_feature} suggests {clinical_finding}',
                'confidence_factors': ['feature_specificity', 'pattern_recognition', 'context_appropriateness']
            },
            'finding_to_diagnosis': {
                'pattern': 'Clinical finding of {clinical_finding} in context of {anatomical_location} indicates {diagnosis}',
                'confidence_factors': ['diagnostic_specificity', 'supporting_features', 'exclusion_criteria']
            },
            'differential_reasoning': {
                'pattern': 'Differential considerations include {alternatives}, but {distinguishing_features} favor {preferred_diagnosis}',
                'confidence_factors': ['discriminating_features', 'clinical_context', 'prevalence']
            }
        }
    
    def identify_anatomical_region(self, visual_context: str, attention_regions: List[Dict]) -> Dict:
        """
        Identify anatomical region from visual context and attention
        
        Args:
            visual_context: Visual description string
            attention_regions: List of attention region dictionaries
            
        Returns:
            Anatomical region information
        """
        identified_regions = []
        confidence_scores = {}
        
        # Analyze visual context for anatomical keywords
        context_lower = visual_context.lower()
        
        for region_name, region_info in self.anatomical_structures.items():
            score = 0
            matched_terms = []
            
            # Check organs
            for organ in region_info['organs']:
                if organ in context_lower:
                    score += 3
                    matched_terms.append(organ)
            
            # Check tissues
            for tissue in region_info['tissues']:
                if tissue in context_lower:
                    score += 2
                    matched_terms.append(tissue)
            
            # Check visual indicators
            for indicator in region_info['visual_indicators']:
                if indicator in context_lower:
                    score += 1
                    matched_terms.append(indicator)
            
            if score > 0:
                identified_regions.append(region_name)
                confidence_scores[region_name] = {
                    'score': score,
                    'matched_terms': matched_terms,
                    'confidence': min(score / 5.0, 1.0)  # Normalize to 0-1
                }
        
        # Sort by confidence
        if identified_regions:
            primary_region = max(identified_regions, key=lambda x: confidence_scores[x]['score'])
            return {
                'primary_region': primary_region,
                'all_regions': identified_regions,
                'confidence_scores': confidence_scores,
                'anatomical_context': self.anatomical_structures.get(primary_region, {})
            }
        else:
            return {
                'primary_region': 'unknown',
                'all_regions': [],
                'confidence_scores': {},
                'anatomical_context': {}
            }
    
    def map_visual_features_to_pathology(self, visual_description: str, 
                                       anatomical_context: Dict) -> Dict:
        """
        Map visual features to pathological processes
        
        Args:
            visual_description: Visual feature description
            anatomical_context: Anatomical context information
            
        Returns:
            Pathology mapping results
        """
        pathology_matches = {}
        description_lower = visual_description.lower()
        
        for pathology_category, subcategories in self.pathology_patterns.items():
            category_score = 0
            matched_features = []
            
            for subtype, features in subcategories.items():
                subtype_score = 0
                
                # Check visual features
                for feature in features.get('visual_features', []):
                    if feature.lower() in description_lower:
                        subtype_score += 2
                        matched_features.append(f"{subtype}:{feature}")
                
                # Check cellular changes
                for change in features.get('cellular_changes', []):
                    if change.lower() in description_lower:
                        subtype_score += 3
                        matched_features.append(f"{subtype}:{change}")
                
                if subtype_score > 0:
                    pathology_matches[f"{pathology_category}_{subtype}"] = {
                        'score': subtype_score,
                        'matched_features': matched_features,
                        'confidence': min(subtype_score / 6.0, 1.0)
                    }
                
                category_score += subtype_score
            
            if category_score > 0:
                pathology_matches[pathology_category] = {
                    'total_score': category_score,
                    'confidence': min(category_score / 10.0, 1.0)
                }
        
        return pathology_matches
    
    def get_diagnostic_reasoning_pattern(self, pathology_type: str, 
                                       anatomical_region: str) -> Dict:
        """
        Get diagnostic reasoning pattern for specific pathology and region
        
        Args:
            pathology_type: Type of pathology identified
            anatomical_region: Anatomical region context
            
        Returns:
            Diagnostic reasoning pattern
        """
        # Get base pathology pattern
        base_pattern = self.pathology_patterns.get(pathology_type, {})
        
        # Get diagnostic criteria if available
        diagnostic_info = {}
        for diagnosis, criteria in self.diagnostic_criteria.items():
            if pathology_type in diagnosis.lower() or anatomical_region in diagnosis.lower():
                diagnostic_info = criteria
                break
        
        # Create reasoning pattern
        reasoning_pattern = {
            'pathology_context': base_pattern,
            'diagnostic_criteria': diagnostic_info,
            'reasoning_steps': [
                {
                    'step': 'visual_observation',
                    'focus': 'Identify key visual features in the image',
                    'template': 'In this {anatomical_region} image, I observe {visual_features}'
                },
                {
                    'step': 'feature_analysis', 
                    'focus': 'Analyze significance of observed features',
                    'template': 'These features of {feature_list} are consistent with {pathological_process}'
                },
                {
                    'step': 'pattern_recognition',
                    'focus': 'Match patterns to known pathology',
                    'template': 'The pattern suggests {pathology_type} based on {diagnostic_criteria}'
                },
                {
                    'step': 'differential_consideration',
                    'focus': 'Consider alternative diagnoses',
                    'template': 'Differential diagnoses include {alternatives}, but {distinguishing_features} favor {primary_diagnosis}'
                },
                {
                    'step': 'conclusion',
                    'focus': 'Synthesize findings into diagnosis',
                    'template': 'Based on the visual evidence, the most likely diagnosis is {final_diagnosis}'
                }
            ]
        }
        
        return reasoning_pattern
    
    def validate_clinical_reasoning(self, reasoning_chain: List[Dict]) -> Dict:
        """
        Validate clinical reasoning chain for medical accuracy
        
        Args:
            reasoning_chain: List of reasoning steps
            
        Returns:
            Validation results
        """
        validation_results = {
            'overall_validity': True,
            'step_validations': [],
            'medical_accuracy_score': 0.0,
            'logical_consistency_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        for i, step in enumerate(reasoning_chain):
            step_validation = {
                'step_number': i + 1,
                'step_type': step.get('type', 'unknown'),
                'is_valid': True,
                'medical_accuracy': 1.0,
                'logical_flow': 1.0,
                'issues': []
            }
            
            # Validate medical terminology
            content = step.get('content', '').lower()
            
            # Check for inappropriate medical claims
            inappropriate_terms = ['definitely', 'certainly', 'absolutely', 'impossible']
            for term in inappropriate_terms:
                if term in content:
                    step_validation['issues'].append(f"Overly definitive language: '{term}'")
                    step_validation['medical_accuracy'] *= 0.8
            
            # Check for logical consistency with previous steps
            if i > 0:
                prev_step = reasoning_chain[i-1]
                # Simple consistency check - could be more sophisticated
                if step.get('confidence', 1.0) > prev_step.get('confidence', 1.0) + 0.2:
                    step_validation['issues'].append("Confidence increase without additional evidence")
                    step_validation['logical_flow'] *= 0.9
            
            # Overall step validity
            if step_validation['issues']:
                step_validation['is_valid'] = False
                validation_results['overall_validity'] = False
            
            validation_results['step_validations'].append(step_validation)
        
        # Compute overall scores
        if validation_results['step_validations']:
            avg_medical = sum(s['medical_accuracy'] for s in validation_results['step_validations']) / len(validation_results['step_validations'])
            avg_logical = sum(s['logical_flow'] for s in validation_results['step_validations']) / len(validation_results['step_validations'])
            
            validation_results['medical_accuracy_score'] = avg_medical
            validation_results['logical_consistency_score'] = avg_logical
        
        return validation_results
