import logging
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import json
import numpy as np

from .medical_knowledge_base import MedicalKnowledgeBase
from .evidence_linker import EvidenceLinker
from .reasoning_templates import ReasoningTemplates

logger = logging.getLogger(__name__)

class ChainOfThoughtGenerator:
    """
    Chain-of-Thought Generator for MedXplain-VQA
    
    Generates structured medical reasoning chains that link visual evidence
    to diagnostic conclusions through step-by-step reasoning process.
    """
    
    def __init__(self, gemini_integration, config):
        """
        Initialize Chain-of-Thought Generator
        
        Args:
            gemini_integration: GeminiIntegration instance
            config: Configuration object
        """
        self.gemini = gemini_integration
        self.config = config
        
        # Initialize components
        self.knowledge_base = MedicalKnowledgeBase(config)
        self.evidence_linker = EvidenceLinker(config)
        self.templates = ReasoningTemplates()
        
        # Reasoning configuration
        self.reasoning_config = {
            'default_flow': config.get('explainability.reasoning.default_flow', 'standard_diagnostic'),
            'confidence_threshold': config.get('explainability.reasoning.confidence_threshold', 0.5),
            'max_reasoning_steps': config.get('explainability.reasoning.max_steps', 8),
            'enable_differential': config.get('explainability.reasoning.enable_differential', True)
        }
        
        # IMPROVED: Confidence calculation parameters
        self.confidence_params = {
            'base_confidence': 0.75,  # Increased from 0.7
            'evidence_weight': 0.3,   # Weight of evidence contribution
            'step_reliability': {     # Step-specific reliability scores
                'visual_observation': 0.90,
                'attention_analysis': 0.85,
                'feature_extraction': 0.82,
                'clinical_correlation': 0.78,
                'pathological_assessment': 0.75,
                'differential_diagnosis': 0.72,
                'diagnostic_reasoning': 0.80,
                'conclusion': 0.85
            },
            'evidence_multipliers': {  # Improved evidence scoring
                'high': 1.0,
                'moderate': 0.95,
                'low': 0.90
            },
            'chain_confidence_method': 'weighted_harmonic_mean'  # Better than multiplicative
        }
        
        logger.info("Chain-of-Thought Generator initialized with improved confidence calculation")
    
    def generate_reasoning_chain(self, image: Image.Image, 
                               reformulated_question: str,
                               blip_answer: str,
                               visual_context: Dict,
                               grad_cam_data: Optional[Dict] = None) -> Dict:
        """
        Generate complete chain-of-thought reasoning
        
        Args:
            image: PIL Image
            reformulated_question: Reformulated question from Phase 3A
            blip_answer: Initial BLIP answer
            visual_context: Visual context from VisualContextExtractor
            grad_cam_data: Grad-CAM attention data (optional)
            
        Returns:
            Complete reasoning chain dictionary
        """
        logger.info("Generating chain-of-thought reasoning")
        
        try:
            # Step 1: Extract and link visual evidence
            visual_evidence = self._extract_visual_evidence(image, grad_cam_data, visual_context)
            
            # Step 2: Identify anatomical and pathological context
            medical_context = self._identify_medical_context(visual_context, visual_evidence)
            
            # Step 3: Determine reasoning flow
            reasoning_flow = self._select_reasoning_flow(reformulated_question, medical_context)
            
            # Step 4: Generate reasoning steps
            reasoning_steps = self._generate_reasoning_steps(
                image, reformulated_question, blip_answer, 
                visual_context, visual_evidence, medical_context, reasoning_flow
            )
            
            # Step 5: Link evidence to steps
            enhanced_steps = self._link_evidence_to_steps(reasoning_steps, visual_evidence)
            
            # Step 6: Create complete reasoning chain
            reasoning_chain = self._create_reasoning_chain(
                enhanced_steps, reasoning_flow, visual_evidence, medical_context
            )
            
            # Step 7: Validate reasoning
            validation_result = self._validate_reasoning_chain(reasoning_chain)
            reasoning_chain['validation'] = validation_result
            
            logger.info(f"Chain-of-thought reasoning generated successfully with confidence: {reasoning_chain.get('reasoning_chain', {}).get('overall_confidence', 0.0):.3f}")
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error generating reasoning chain: {e}")
            
            # Return error result with basic structure
            return {
                'success': False,
                'error': str(e),
                'reasoning_chain': {
                    'steps': [{
                        'type': 'error',
                        'content': f"Unable to generate reasoning: {str(e)}",
                        'confidence': 0.0
                    }],
                    'overall_confidence': 0.0
                }
            }
    
    def _extract_visual_evidence(self, image: Image.Image, 
                                grad_cam_data: Optional[Dict],
                                visual_context: Dict) -> Dict:
        """Extract visual evidence from all sources"""
        logger.debug("Extracting visual evidence")
        
        # Use evidence linker to extract visual evidence
        visual_evidence = self.evidence_linker.extract_visual_evidence(
            image, grad_cam_data or {}, visual_context
        )
        
        return visual_evidence
    
    def _identify_medical_context(self, visual_context: Dict, 
                                 visual_evidence: Dict) -> Dict:
        """Identify medical context using knowledge base"""
        logger.debug("Identifying medical context")
        
        # Extract anatomical context
        attention_regions = visual_evidence.get('attention_evidence', {}).get('primary_regions', [])
        anatomical_info = self.knowledge_base.identify_anatomical_region(
            visual_context.get('visual_description', ''),
            attention_regions
        )
        
        # Map visual features to pathology
        pathology_mapping = self.knowledge_base.map_visual_features_to_pathology(
            visual_context.get('visual_description', ''),
            anatomical_info
        )
        
        medical_context = {
            'anatomical_info': anatomical_info,
            'pathology_mapping': pathology_mapping,
            'primary_region': anatomical_info.get('primary_region', 'unknown'),
            'primary_pathology': self._identify_primary_pathology(pathology_mapping)
        }
        
        return medical_context
    
    def _identify_primary_pathology(self, pathology_mapping: Dict) -> str:
        """Identify primary pathology from mapping results"""
        if not pathology_mapping:
            return 'unknown'
        
        # Find pathology with highest confidence
        max_confidence = 0
        primary_pathology = 'unknown'
        
        for pathology, info in pathology_mapping.items():
            if isinstance(info, dict) and 'confidence' in info:
                if info['confidence'] > max_confidence:
                    max_confidence = info['confidence']
                    primary_pathology = pathology
        
        return primary_pathology
    
    def _select_reasoning_flow(self, question: str, medical_context: Dict) -> str:
        """Select appropriate reasoning flow based on question and context"""
        question_lower = question.lower()
        primary_pathology = medical_context.get('primary_pathology', 'unknown')
        
        # Select flow based on question type and pathology
        if 'differential' in question_lower or 'diagnosis' in question_lower:
            return 'comparative_analysis'
        elif 'pathology' in question_lower or 'tissue' in question_lower:
            return 'pathology_focused'
        elif 'attention' in question_lower or 'focus' in question_lower:
            return 'attention_guided'
        elif primary_pathology != 'unknown':
            return 'pathology_focused'
        else:
            return self.reasoning_config['default_flow']
    
    def _generate_reasoning_steps(self, image: Image.Image,
                                 question: str,
                                 blip_answer: str,
                                 visual_context: Dict,
                                 visual_evidence: Dict,
                                 medical_context: Dict,
                                 reasoning_flow: str) -> List[Dict]:
        """Generate individual reasoning steps"""
        logger.debug(f"Generating reasoning steps using {reasoning_flow} flow")
        
        reasoning_steps = []
        
        # Get flow template
        flow_info = self.templates.get_reasoning_flow(reasoning_flow)
        expected_steps = flow_info['steps']
        
        for step_type in expected_steps:
            step_data = self._generate_step_data(
                step_type, question, blip_answer, visual_context, 
                visual_evidence, medical_context
            )
            
            # Generate step content using Gemini
            step_content = self._generate_step_content_with_gemini(
                step_type, step_data, question, blip_answer
            )
            
            # IMPROVED: Calculate step confidence with new method
            step_confidence = self._calculate_step_confidence_improved(
                step_type, step_data, visual_evidence, medical_context
            )
            
            reasoning_step = {
                'type': step_type,
                'content': step_content,
                'data': step_data,
                'confidence': step_confidence
            }
            
            reasoning_steps.append(reasoning_step)
        
        return reasoning_steps
    
    def _generate_step_data(self, step_type: str, question: str, blip_answer: str,
                           visual_context: Dict, visual_evidence: Dict, 
                           medical_context: Dict) -> Dict:
        """Generate data for specific reasoning step"""
        base_data = {
            'question': question,
            'blip_answer': blip_answer,
            'visual_description': visual_context.get('visual_description', ''),
            'anatomical_context': visual_context.get('anatomical_context', ''),
            'primary_region': medical_context.get('primary_region', 'unknown')
        }
        
        if step_type == 'visual_observation':
            base_data.update({
                'image_type': 'medical',
                'anatomical_region': medical_context.get('primary_region', 'tissue'),
                'visual_features': visual_context.get('visual_description', 'various features'),
                'additional_details': f"Image dimensions and quality appear suitable for analysis"
            })
        
        elif step_type == 'attention_analysis':
            attention_evidence = visual_evidence.get('attention_evidence', {})
            primary_regions = attention_evidence.get('primary_regions', [])
            
            if primary_regions:
                focus_desc = f"primary focus on {len(primary_regions)} high-attention regions"
                attention_pattern = "concentrated"
            else:
                focus_desc = "distributed attention across multiple areas"
                attention_pattern = "distributed"
            
            base_data.update({
                'attention_pattern': attention_pattern,
                'focus_description': focus_desc,
                'attention_significance': "indicating key diagnostic features"
            })
        
        elif step_type == 'feature_extraction':
            feature_evidence = visual_evidence.get('feature_evidence', {})
            visual_descriptors = feature_evidence.get('visual_descriptors', [])
            pathological_features = feature_evidence.get('pathological_features', [])
            
            base_data.update({
                'feature_list': ', '.join(visual_descriptors + pathological_features) or 'visual characteristics',
                'characteristics': 'distinct morphological patterns',
                'spatial_distribution': 'throughout the visible region'
            })
        
        elif step_type == 'clinical_correlation':
            base_data.update({
                'visual_findings': visual_context.get('visual_description', 'observed features'),
                'clinical_interpretation': medical_context.get('primary_pathology', 'pathological changes'),
                'supporting_evidence': f"based on {medical_context.get('primary_region', 'anatomical')} context"
            })
        
        elif step_type == 'pathological_assessment':
            primary_pathology = medical_context.get('primary_pathology', 'pathological changes')
            base_data.update({
                'pathology_type': primary_pathology,
                'pathological_changes': 'cellular and tissue alterations',
                'severity_assessment': 'requiring further clinical correlation'
            })
        
        elif step_type == 'differential_diagnosis':
            primary_pathology = medical_context.get('primary_pathology', 'primary condition')
            base_data.update({
                'alternative_diagnoses': 'other potential conditions',
                'distinguishing_features': 'specific visual characteristics',
                'preferred_diagnosis': primary_pathology
            })
        
        elif step_type == 'diagnostic_reasoning':
            base_data.update({
                'evidence_summary': 'visual and analytical evidence',
                'diagnosis': blip_answer or 'diagnostic findings',
                'confidence_level': 'moderate to high',
                'reasoning_rationale': 'based on systematic visual analysis'
            })
        
        elif step_type == 'conclusion':
            base_data.update({
                'anatomical_region': medical_context.get('primary_region', 'tissue'),  # FIX: Add missing field
                'key_findings': visual_context.get('visual_description', 'key observations'),
                'final_diagnosis': blip_answer or 'analytical findings',
                'clinical_implications': 'relevant for clinical assessment'
            })
        
        return base_data
    
    def _generate_step_content_with_gemini(self, step_type: str, step_data: Dict,
                                          question: str, blip_answer: str) -> str:
        """Generate step content using Gemini LLM"""
        
        # Create prompt for Gemini
        prompt = f"""
        Generate a medical reasoning step for chain-of-thought analysis.
        
        Step Type: {step_type}
        Question: {question}
        Initial Answer: {blip_answer}
        
        Context:
        - Visual Description: {step_data.get('visual_description', '')}
        - Anatomical Context: {step_data.get('anatomical_context', '')}
        - Primary Region: {step_data.get('primary_region', '')}
        
        Requirements:
        - Write a single, clear reasoning step appropriate for {step_type}
        - Use medical terminology appropriately
        - Reference visual evidence when relevant
        - Keep it concise but informative (2-3 sentences)
        - Maintain clinical objectivity
        
        Generated reasoning step:
        """
        
        try:
            response = self.gemini.model.generate_content(
                prompt,
                generation_config=self.gemini.generation_config
            )
            
            generated_content = response.text.strip()
            
            # Clean up the response
            if "Generated reasoning step:" in generated_content:
                generated_content = generated_content.split("Generated reasoning step:")[-1].strip()
            
            return generated_content
            
        except Exception as e:
            logger.error(f"Error generating step content with Gemini: {e}")
            
            # Fallback to template-based generation
            template_info = self.templates.get_step_template(step_type)
            try:
                return template_info['template'].format(**step_data)
            except:
                return f"Analysis for {step_type}: {step_data.get('visual_description', 'visual findings observed')}"
    
    def _calculate_step_confidence_improved(self, step_type: str, step_data: Dict, 
                                          visual_evidence: Dict, medical_context: Dict) -> float:
        """
        IMPROVED: Calculate confidence for reasoning step with better scoring
        
        Args:
            step_type: Type of reasoning step
            step_data: Step data dictionary
            visual_evidence: Visual evidence dictionary
            medical_context: Medical context dictionary
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base reliability for this step type
        base_reliability = self.confidence_params['step_reliability'].get(step_type, 0.75)
        
        # Evidence quality assessment
        evidence_quality = self._assess_evidence_quality(visual_evidence)
        
        # Medical context strength
        context_strength = self._assess_context_strength(medical_context)
        
        # Step-specific confidence factors
        step_factors = self._get_step_specific_factors(step_type, step_data, visual_evidence)
        
        # Combine factors using weighted formula
        evidence_weight = self.confidence_params['evidence_weight']
        
        confidence = (
            base_reliability * 0.4 +                    # Base step reliability (40%)
            evidence_quality * evidence_weight +        # Evidence contribution (30%)
            context_strength * 0.2 +                    # Medical context (20%)
            step_factors * 0.1                         # Step-specific factors (10%)
        )
        
        # Ensure confidence is within reasonable bounds
        confidence = max(0.5, min(confidence, 0.95))  # Clamp between 50% and 95%
        
        logger.debug(f"Step {step_type} confidence: {confidence:.3f} (base: {base_reliability}, evidence: {evidence_quality:.3f}, context: {context_strength:.3f})")
        
        return confidence
    
    def _assess_evidence_quality(self, visual_evidence: Dict) -> float:
        """Assess overall quality of visual evidence"""
        evidence_summary = visual_evidence.get('summary', {})
        
        # Base quality from evidence summary
        confidence_level = evidence_summary.get('confidence_level', 'moderate')
        base_quality = self.confidence_params['evidence_multipliers'].get(confidence_level, 0.95)
        
        # Factor in number of evidence sources
        evidence_sources = evidence_summary.get('total_evidence_sources', 0)
        source_bonus = min(evidence_sources * 0.05, 0.15)  # Max 15% bonus for multiple sources
        
        # Factor in attention evidence quality
        attention_evidence = visual_evidence.get('attention_evidence', {})
        attention_bonus = 0.0
        if attention_evidence.get('primary_regions'):
            primary_regions = attention_evidence['primary_regions']
            if primary_regions:
                max_attention_score = max(region.get('score', 0) for region in primary_regions)
                attention_bonus = max_attention_score * 0.1  # Max 10% bonus for strong attention
        
        total_quality = base_quality + source_bonus + attention_bonus
        return min(total_quality, 1.0)
    
    def _assess_context_strength(self, medical_context: Dict) -> float:
        """Assess strength of medical context"""
        anatomical_info = medical_context.get('anatomical_info', {})
        pathology_mapping = medical_context.get('pathology_mapping', {})
        
        # Anatomical context strength
        anatomical_strength = 0.5  # Base
        if anatomical_info.get('primary_region') != 'unknown':
            anatomical_strength = 0.7
            
            # Bonus for confidence scores
            confidence_scores = anatomical_info.get('confidence_scores', {})
            if confidence_scores:
                max_confidence = max(info.get('confidence', 0) for info in confidence_scores.values())
                anatomical_strength += max_confidence * 0.2
        
        # Pathology context strength
        pathology_strength = 0.5  # Base
        if pathology_mapping:
            pathology_confidences = []
            for info in pathology_mapping.values():
                if isinstance(info, dict) and 'confidence' in info:
                    pathology_confidences.append(info['confidence'])
            
            if pathology_confidences:
                avg_pathology_confidence = np.mean(pathology_confidences)
                pathology_strength = 0.5 + avg_pathology_confidence * 0.4
        
        # Combine anatomical and pathology strengths
        overall_strength = (anatomical_strength + pathology_strength) / 2
        return min(overall_strength, 1.0)
    
    def _get_step_specific_factors(self, step_type: str, step_data: Dict, 
                                  visual_evidence: Dict) -> float:
        """Get step-specific confidence factors"""
        if step_type == 'visual_observation':
            # High confidence for direct visual observations
            return 0.9
        
        elif step_type == 'attention_analysis':
            # Confidence based on attention evidence quality
            attention_evidence = visual_evidence.get('attention_evidence', {})
            if attention_evidence.get('primary_regions'):
                return 0.85
            return 0.7
        
        elif step_type == 'feature_extraction':
            # Confidence based on feature evidence
            feature_evidence = visual_evidence.get('feature_evidence', {})
            feature_count = len(feature_evidence.get('visual_descriptors', []) + 
                             feature_evidence.get('pathological_features', []))
            return min(0.6 + feature_count * 0.05, 0.9)
        
        elif step_type in ['clinical_correlation', 'pathological_assessment']:
            # Moderate confidence for interpretive steps
            return 0.75
        
        elif step_type == 'differential_diagnosis':
            # Lower confidence for differential reasoning
            return 0.7
        
        elif step_type == 'diagnostic_reasoning':
            # Confidence based on evidence synthesis
            return 0.8
        
        elif step_type == 'conclusion':
            # High confidence for final synthesis
            return 0.85
        
        return 0.75  # Default
    
    def _link_evidence_to_steps(self, reasoning_steps: List[Dict], 
                               visual_evidence: Dict) -> List[Dict]:
        """Link visual evidence to reasoning steps"""
        logger.debug("Linking evidence to reasoning steps")
        
        enhanced_steps = []
        
        for step in reasoning_steps:
            enhanced_step = self.evidence_linker.link_evidence_to_reasoning_step(
                step, visual_evidence
            )
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps
    
    def _create_reasoning_chain(self, enhanced_steps: List[Dict], 
                               reasoning_flow: str,
                               visual_evidence: Dict,
                               medical_context: Dict) -> Dict:
        """Create complete reasoning chain"""
        logger.debug("Creating complete reasoning chain")
        
        # Use templates to create structured chain but with improved confidence calculation
        steps_data = [step['data'] for step in enhanced_steps]
        template_chain = self.templates.create_reasoning_chain(reasoning_flow, steps_data)
        
        # Enhance with our generated content
        for i, enhanced_step in enumerate(enhanced_steps):
            if i < len(template_chain['steps']):
                template_chain['steps'][i].update({
                    'content': enhanced_step['content'],
                    'confidence': enhanced_step['confidence'],
                    'evidence_links': enhanced_step.get('evidence_links', {}),
                    'step_data': enhanced_step['data']
                })
        
        # IMPROVED: Calculate overall confidence with better method
        template_chain['overall_confidence'] = self._calculate_chain_confidence_improved(
            enhanced_steps, reasoning_flow
        )
        
        # Add metadata
        reasoning_chain = {
            'success': True,
            'reasoning_chain': template_chain,
            'metadata': {
                'flow_type': reasoning_flow,
                'total_steps': len(enhanced_steps),
                'visual_evidence_summary': visual_evidence.get('summary', {}),
                'medical_context': medical_context,
                'generation_timestamp': self._get_timestamp(),
                'confidence_method': self.confidence_params['chain_confidence_method']
            }
        }
        
        return reasoning_chain
    
    def _calculate_chain_confidence_improved(self, steps: List[Dict], 
                                           reasoning_flow: str) -> float:
        """
        IMPROVED: Calculate overall confidence for reasoning chain
        
        Uses weighted harmonic mean instead of multiplicative for better results
        """
        if not steps:
            return 0.0
        
        step_confidences = [step.get('confidence', 0.5) for step in steps]
        method = self.confidence_params['chain_confidence_method']
        
        if method == 'weighted_harmonic_mean':
            # Weighted harmonic mean - more robust than multiplicative
            weights = self._get_step_weights(steps, reasoning_flow)
            
            # Calculate weighted harmonic mean
            if len(step_confidences) == len(weights):
                weighted_reciprocals = [w / c for w, c in zip(weights, step_confidences) if c > 0]
                sum_weights = sum(weights)
                
                if weighted_reciprocals and sum_weights > 0:
                    harmonic_mean = sum_weights / sum(weighted_reciprocals)
                    # Apply small confidence boost for complete chains
                    completion_bonus = 0.05 if len(steps) >= 5 else 0.0
                    return min(harmonic_mean + completion_bonus, 0.95)
        
        elif method == 'weighted_geometric_mean':
            # Weighted geometric mean - better than arithmetic for confidence
            weights = self._get_step_weights(steps, reasoning_flow)
            
            if len(step_confidences) == len(weights):
                weighted_product = 1.0
                total_weight = sum(weights)
                
                for conf, weight in zip(step_confidences, weights):
                    weighted_product *= (conf ** (weight / total_weight))
                
                return min(weighted_product, 0.95)
        
        elif method == 'confidence_cascade':
            # Confidence cascade - each step builds on previous
            cascade_confidence = step_confidences[0] if step_confidences else 0.5
            
            for i in range(1, len(step_confidences)):
                # Each step contributes but with diminishing returns
                contribution = step_confidences[i] * (0.9 ** i)  # Diminishing factor
                cascade_confidence = (cascade_confidence + contribution) / 2
            
            return min(cascade_confidence, 0.95)
        
        # Fallback: Weighted average (safer than multiplicative)
        weights = self._get_step_weights(steps, reasoning_flow)
        if len(step_confidences) == len(weights):
            weighted_sum = sum(c * w for c, w in zip(step_confidences, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            return np.mean(step_confidences)
    
    def _get_step_weights(self, steps: List[Dict], reasoning_flow: str) -> List[float]:
        """Get weights for different reasoning steps based on flow type"""
        step_types = [step.get('type', 'unknown') for step in steps]
        
        # Default weights
        default_weights = {
            'visual_observation': 1.2,     # High weight for direct observations
            'attention_analysis': 1.1,     # High weight for attention analysis
            'feature_extraction': 1.0,     # Standard weight
            'clinical_correlation': 1.3,   # Higher weight for clinical insights
            'pathological_assessment': 1.2, # High weight for pathology
            'differential_diagnosis': 0.9,  # Lower weight for differential
            'diagnostic_reasoning': 1.4,    # Highest weight for final reasoning
            'conclusion': 1.3              # High weight for conclusions
        }
        
        # Flow-specific weight adjustments
        if reasoning_flow == 'pathology_focused':
            default_weights['pathological_assessment'] = 1.5
            default_weights['clinical_correlation'] = 1.4
        elif reasoning_flow == 'attention_guided':
            default_weights['attention_analysis'] = 1.4
            default_weights['visual_observation'] = 1.3
        elif reasoning_flow == 'comparative_analysis':
            default_weights['differential_diagnosis'] = 1.2
            default_weights['diagnostic_reasoning'] = 1.5
        
        # Create weight list
        weights = [default_weights.get(step_type, 1.0) for step_type in step_types]
        
        return weights
    
    def _validate_reasoning_chain(self, reasoning_chain: Dict) -> Dict:
        """Validate generated reasoning chain"""
        logger.debug("Validating reasoning chain")
        
        chain_data = reasoning_chain.get('reasoning_chain', {})
        
        # Use templates validation
        template_validation = self.templates.validate_reasoning_chain(chain_data)
        
        # Add medical knowledge validation
        steps = chain_data.get('steps', [])
        medical_validation = self.knowledge_base.validate_clinical_reasoning(steps)
        
        # IMPROVED: Confidence-aware validation
        overall_confidence = chain_data.get('overall_confidence', 0.0)
        confidence_validity = overall_confidence >= 0.5  # Minimum acceptable confidence
        
        # Combine validations
        combined_validation = {
            'template_validation': template_validation,
            'medical_validation': medical_validation,
            'confidence_validation': {
                'confidence_level': overall_confidence,
                'meets_threshold': confidence_validity,
                'confidence_category': self._categorize_confidence(overall_confidence)
            },
            'overall_validity': (template_validation['is_valid'] and 
                               medical_validation['overall_validity'] and 
                               confidence_validity),
            'combined_score': (template_validation.get('completeness_score', 0) + 
                             template_validation.get('consistency_score', 0) +
                             medical_validation.get('medical_accuracy_score', 0) +
                             medical_validation.get('logical_consistency_score', 0) +
                             overall_confidence) / 5  # Include confidence in overall score
        }
        
        return combined_validation
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.65:
            return 'moderate-high'
        elif confidence >= 0.5:
            return 'moderate'
        elif confidence >= 0.35:
            return 'low-moderate'
        else:
            return 'low'
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_reasoning_chain(self, reasoning_chain: Dict, output_path: str):
        """
        Save reasoning chain to file
        
        Args:
            reasoning_chain: Complete reasoning chain
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_chain, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reasoning chain saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving reasoning chain: {e}")
