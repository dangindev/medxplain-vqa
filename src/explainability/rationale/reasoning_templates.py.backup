import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class ReasoningTemplates:
    """
    Templates for structured medical reasoning chains
    Provides standardized formats for different types of reasoning steps
    """
    
    def __init__(self):
        """Initialize reasoning templates"""
        self.step_templates = self._init_step_templates()
        self.reasoning_flows = self._init_reasoning_flows()
        self.evidence_templates = self._init_evidence_templates()
        
        logger.info("Reasoning Templates initialized")
    
    def _init_step_templates(self) -> Dict:
        """Initialize templates for individual reasoning steps"""
        return {
            'visual_observation': {
                'template': "In this {image_type} image of {anatomical_region}, I observe {visual_features}. {additional_details}",
                'required_fields': ['image_type', 'anatomical_region', 'visual_features'],
                'optional_fields': ['additional_details'],
                'confidence_factors': ['feature_clarity', 'image_quality', 'anatomical_certainty']
            },
            
            'attention_analysis': {
                'template': "The model's attention is {attention_pattern} with {focus_description}. {attention_significance}",
                'required_fields': ['attention_pattern', 'focus_description'],
                'optional_fields': ['attention_significance'],
                'confidence_factors': ['attention_strength', 'spatial_relevance', 'pattern_consistency']
            },
            
            'feature_extraction': {
                'template': "Key visual features include {feature_list}. These features exhibit {characteristics} and are located {spatial_distribution}.",
                'required_fields': ['feature_list', 'characteristics'],
                'optional_fields': ['spatial_distribution'],
                'confidence_factors': ['feature_specificity', 'visibility', 'diagnostic_relevance']
            },
            
            'clinical_correlation': {
                'template': "The observed {visual_findings} are consistent with {clinical_interpretation}. {supporting_evidence}",
                'required_fields': ['visual_findings', 'clinical_interpretation'],
                'optional_fields': ['supporting_evidence'],
                'confidence_factors': ['correlation_strength', 'medical_evidence', 'pattern_match']
            },
            
            'pathological_assessment': {
                'template': "The pathological features suggest {pathology_type} characterized by {pathological_changes}. {severity_assessment}",
                'required_fields': ['pathology_type', 'pathological_changes'],
                'optional_fields': ['severity_assessment'],
                'confidence_factors': ['pathology_specificity', 'feature_consistency', 'diagnostic_confidence']
            },
            
            'differential_diagnosis': {
                'template': "Differential considerations include {alternative_diagnoses}. However, {distinguishing_features} favor {preferred_diagnosis}.",
                'required_fields': ['alternative_diagnoses', 'distinguishing_features', 'preferred_diagnosis'],
                'optional_fields': [],
                'confidence_factors': ['diagnostic_specificity', 'exclusion_strength', 'differential_clarity']
            },
            
            'diagnostic_reasoning': {
                'template': "Based on {evidence_summary}, the findings support {diagnosis} with {confidence_level} confidence. {reasoning_rationale}",
                'required_fields': ['evidence_summary', 'diagnosis', 'confidence_level'],
                'optional_fields': ['reasoning_rationale'],
                'confidence_factors': ['evidence_strength', 'logical_consistency', 'medical_validity']
            },
            
            'conclusion': {
                'template': "In conclusion, this {anatomical_region} image demonstrates {key_findings} consistent with {final_diagnosis}. {clinical_implications}",
                'required_fields': ['anatomical_region', 'key_findings', 'final_diagnosis'],
                'optional_fields': ['clinical_implications'],
                'confidence_factors': ['conclusion_strength', 'evidence_synthesis', 'diagnostic_certainty']
            }
        }
    
    def _init_reasoning_flows(self) -> Dict:
        """Initialize different reasoning flow patterns"""
        return {
            'standard_diagnostic': {
                'description': 'Standard diagnostic reasoning flow',
                'steps': [
                    'visual_observation',
                    'attention_analysis', 
                    'feature_extraction',
                    'clinical_correlation',
                    'diagnostic_reasoning',
                    'conclusion'
                ],
                'confidence_propagation': 'weighted_harmonic_mean'  # IMPROVED
            },
            
            'pathology_focused': {
                'description': 'Pathology-focused reasoning for tissue analysis',
                'steps': [
                    'visual_observation',
                    'feature_extraction',
                    'pathological_assessment',
                    'clinical_correlation',
                    'differential_diagnosis',
                    'conclusion'
                ],
                'confidence_propagation': 'weighted_geometric_mean'  # IMPROVED
            },
            
            'attention_guided': {
                'description': 'Attention-guided reasoning emphasizing model focus',
                'steps': [
                    'visual_observation',
                    'attention_analysis',
                    'feature_extraction',
                    'clinical_correlation',
                    'diagnostic_reasoning',
                    'conclusion'
                ],
                'confidence_propagation': 'confidence_cascade'  # IMPROVED
            },
            
            'comparative_analysis': {
                'description': 'Comparative analysis with differential diagnosis',
                'steps': [
                    'visual_observation',
                    'feature_extraction', 
                    'clinical_correlation',
                    'differential_diagnosis',
                    'diagnostic_reasoning',
                    'conclusion'
                ],
                'confidence_propagation': 'weighted_harmonic_mean'  # IMPROVED
            }
        }
    
    def _init_evidence_templates(self) -> Dict:
        """Initialize templates for evidence citation"""
        return {
            'visual_evidence': {
                'template': "Visual evidence: {evidence_description} (confidence: {confidence})",
                'citation_format': "[Visual: {location}]"
            },
            
            'attention_evidence': {
                'template': "Attention evidence: {attention_description} (strength: {strength})",
                'citation_format': "[Attention: {region}]"
            },
            
            'spatial_evidence': {
                'template': "Spatial evidence: {spatial_description} (relevance: {relevance})",
                'citation_format': "[Spatial: {coordinates}]"
            },
            
            'clinical_evidence': {
                'template': "Clinical evidence: {clinical_description} (validity: {validity})",
                'citation_format': "[Clinical: {source}]"
            },
            
            'pattern_evidence': {
                'template': "Pattern evidence: {pattern_description} (match: {match_score})",
                'citation_format': "[Pattern: {pattern_type}]"
            }
        }
    
    def get_step_template(self, step_type: str) -> Dict:
        """Get template for specific reasoning step type"""
        return self.step_templates.get(step_type, {
            'template': "Analysis step: {content}",
            'required_fields': ['content'],
            'optional_fields': [],
            'confidence_factors': ['general_confidence']
        })
    
    def get_reasoning_flow(self, flow_type: str) -> Dict:
        """Get reasoning flow template"""
        return self.reasoning_flows.get(flow_type, self.reasoning_flows['standard_diagnostic'])
    
    def format_reasoning_step(self, step_type: str, step_data: Dict) -> Dict:
        """Format a reasoning step using appropriate template"""
        template_info = self.get_step_template(step_type)
        template = template_info['template']
        required_fields = template_info['required_fields']
        optional_fields = template_info['optional_fields']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in step_data]
        if missing_fields:
            logger.warning(f"Missing required fields for {step_type}: {missing_fields}")
            # Provide default values for missing fields
            for field in missing_fields:
                step_data[field] = f"[{field}]"
        
        # Provide default values for optional fields
        for field in optional_fields:
            if field not in step_data:
                step_data[field] = ""
        
        # Format template
        try:
            formatted_content = template.format(**step_data)
        except KeyError as e:
            logger.error(f"Template formatting error for {step_type}: {e}")
            formatted_content = f"Error formatting {step_type} step"
        
        # Create formatted step
        formatted_step = {
            'type': step_type,
            'content': formatted_content,
            'template_used': template,
            'input_data': step_data,
            'confidence_factors': template_info['confidence_factors']
        }
        
        return formatted_step
    
    def create_reasoning_chain(self, flow_type: str, steps_data: List[Dict]) -> Dict:
        """Create complete reasoning chain using specified flow"""
        flow_info = self.get_reasoning_flow(flow_type)
        expected_steps = flow_info['steps']
        
        reasoning_chain = {
            'flow_type': flow_type,
            'flow_description': flow_info['description'],
            'steps': [],
            'confidence_propagation': flow_info['confidence_propagation'],
            'overall_confidence': 0.0
        }
        
        # Process each step
        for i, step_type in enumerate(expected_steps):
            if i < len(steps_data):
                step_data = steps_data[i]
                formatted_step = self.format_reasoning_step(step_type, step_data)
                
                # Add step number and flow position
                formatted_step['step_number'] = i + 1
                formatted_step['flow_position'] = f"{i + 1}/{len(expected_steps)}"
                
                reasoning_chain['steps'].append(formatted_step)
            else:
                logger.warning(f"No data provided for step {step_type} in {flow_type} flow")
        
        # Note: Overall confidence will be calculated by ChainOfThoughtGenerator
        # using the improved confidence calculation methods
        
        return reasoning_chain
    
    def add_evidence_citations(self, reasoning_step: Dict, 
                              evidence_links: List[Dict]) -> Dict:
        """Add evidence citations to reasoning step"""
        step_with_evidence = reasoning_step.copy()
        citations = []
        
        for evidence in evidence_links:
            evidence_type = evidence.get('type', 'unknown')
            template_info = self.evidence_templates.get(f"{evidence_type}_evidence", 
                                                       self.evidence_templates['visual_evidence'])
            
            # Format evidence description
            evidence_description = evidence.get('description', 'Evidence available')
            confidence = evidence.get('confidence', evidence.get('relevance', 'moderate'))
            
            # Create citation
            citation = template_info['citation_format'].format(
                location=evidence.get('location', 'unspecified'),
                region=evidence.get('region', 'unspecified'),
                coordinates=evidence.get('coordinates', 'unspecified'),
                source=evidence.get('source', 'analysis'),
                pattern_type=evidence.get('pattern_type', 'unspecified')
            )
            
            citations.append({
                'citation': citation,
                'evidence_type': evidence_type,
                'description': evidence_description,
                'confidence': confidence
            })
        
        # Add citations to step
        step_with_evidence['evidence_citations'] = citations
        
        # Append citations to content
        if citations:
            citation_text = " " + " ".join([c['citation'] for c in citations])
            step_with_evidence['content'] += citation_text
        
        return step_with_evidence
    
    def validate_reasoning_chain(self, reasoning_chain: Dict) -> Dict:
        """Validate reasoning chain for completeness and consistency"""
        validation = {
            'is_valid': True,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        steps = reasoning_chain.get('steps', [])
        flow_type = reasoning_chain.get('flow_type', 'unknown')
        
        # Check completeness
        expected_flow = self.get_reasoning_flow(flow_type)
        expected_steps = expected_flow['steps']
        
        if len(steps) < len(expected_steps):
            validation['issues'].append(f"Incomplete reasoning chain: {len(steps)}/{len(expected_steps)} steps")
            validation['is_valid'] = False
        
        validation['completeness_score'] = len(steps) / len(expected_steps) if expected_steps else 0
        
        # IMPROVED: Check consistency with better confidence awareness
        consistency_issues = 0
        confidence_drops = 0
        
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            # Check confidence consistency
            current_conf = current_step.get('confidence', 0.5)
            previous_conf = previous_step.get('confidence', 0.5)
            
            # Allow reasonable confidence variations
            confidence_drop = previous_conf - current_conf
            
            if confidence_drop > 0.2:  # Significant confidence drop
                confidence_drops += 1
                if confidence_drop > 0.3:  # Major confidence drop
                    consistency_issues += 1
                    validation['issues'].append(f"Step {i+1}: Major confidence drop ({confidence_drop:.2f})")
            elif current_conf > previous_conf + 0.3:  # Unreasonable confidence increase
                consistency_issues += 1
                validation['issues'].append(f"Step {i+1}: Confidence increase without justification")
        
        # Calculate consistency score
        if len(steps) > 1:
            max_issues = len(steps) - 1
            validation['consistency_score'] = max(0, 1.0 - (consistency_issues / max_issues))
        else:
            validation['consistency_score'] = 1.0
        
        # Overall validity
        if validation['consistency_score'] < 0.6:
            validation['is_valid'] = False
        
        # Generate improved suggestions
        if validation['completeness_score'] < 1.0:
            validation['suggestions'].append("Consider adding missing reasoning steps")
        
        if validation['consistency_score'] < 0.8:
            validation['suggestions'].append("Review confidence assignments for logical consistency")
        
        if confidence_drops > 0:
            validation['suggestions'].append("Investigate confidence drops and strengthen evidence support")
        
        return validation
