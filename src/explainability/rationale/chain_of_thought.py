import logging
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import json

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
        
        logger.info("Chain-of-Thought Generator initialized")
    
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
            
            logger.info("Chain-of-thought reasoning generated successfully")
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
            
            reasoning_step = {
                'type': step_type,
                'content': step_content,
                'data': step_data,
                'confidence': self._calculate_step_confidence(step_type, step_data, visual_evidence)
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
    
    def _calculate_step_confidence(self, step_type: str, step_data: Dict, 
                                  visual_evidence: Dict) -> float:
        """Calculate confidence for reasoning step"""
        base_confidence = 0.7  # Base confidence
        
        # Adjust based on step type
        step_confidence_weights = {
            'visual_observation': 0.9,  # High confidence for direct observations
            'attention_analysis': 0.8,   # High confidence for attention analysis
            'feature_extraction': 0.8,   # High confidence for feature identification
            'clinical_correlation': 0.7,  # Moderate confidence for correlations
            'pathological_assessment': 0.6, # Lower confidence for assessments
            'differential_diagnosis': 0.6,  # Lower confidence for differentials
            'diagnostic_reasoning': 0.7,   # Moderate confidence for reasoning
            'conclusion': 0.8             # High confidence for conclusions
        }
        
        step_weight = step_confidence_weights.get(step_type, 0.7)
        
        # Adjust based on evidence quality
        evidence_summary = visual_evidence.get('summary', {})
        evidence_confidence = evidence_summary.get('confidence_level', 'moderate')
        
        evidence_multipliers = {
            'high': 1.0,
            'moderate': 0.9,
            'low': 0.8
        }
        
        evidence_multiplier = evidence_multipliers.get(evidence_confidence, 0.9)
        
        final_confidence = base_confidence * step_weight * evidence_multiplier
        return min(final_confidence, 1.0)
    
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
        
        # Use templates to create structured chain
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
        
        # Add metadata
        reasoning_chain = {
            'success': True,
            'reasoning_chain': template_chain,
            'metadata': {
                'flow_type': reasoning_flow,
                'total_steps': len(enhanced_steps),
                'visual_evidence_summary': visual_evidence.get('summary', {}),
                'medical_context': medical_context,
                'generation_timestamp': self._get_timestamp()
            }
        }
        
        return reasoning_chain
    
    def _validate_reasoning_chain(self, reasoning_chain: Dict) -> Dict:
        """Validate generated reasoning chain"""
        logger.debug("Validating reasoning chain")
        
        chain_data = reasoning_chain.get('reasoning_chain', {})
        
        # Use templates validation
        template_validation = self.templates.validate_reasoning_chain(chain_data)
        
        # Add medical knowledge validation
        steps = chain_data.get('steps', [])
        medical_validation = self.knowledge_base.validate_clinical_reasoning(steps)
        
        # Combine validations
        combined_validation = {
            'template_validation': template_validation,
            'medical_validation': medical_validation,
            'overall_validity': template_validation['is_valid'] and medical_validation['overall_validity'],
            'combined_score': (template_validation.get('completeness_score', 0) + 
                             template_validation.get('consistency_score', 0) +
                             medical_validation.get('medical_accuracy_score', 0) +
                             medical_validation.get('logical_consistency_score', 0)) / 4
        }
        
        return combined_validation
    
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
