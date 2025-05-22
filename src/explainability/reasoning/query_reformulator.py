import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import json

logger = logging.getLogger(__name__)

class QueryReformulator:
    """
    Reformulate clinical questions to be self-contained and grounded in image content
    
    This class implements the core functionality described in the abstract:
    "viết lại mỗi câu hỏi lâm sàng thành công thức tự chứa được grounded rõ ràng trong hình ảnh liên quan"
    """
    
    def __init__(self, gemini_integration, visual_context_extractor, config):
        """
        Initialize Query Reformulator
        
        Args:
            gemini_integration: GeminiIntegration instance
            visual_context_extractor: VisualContextExtractor instance  
            config: Configuration object
        """
        self.gemini = gemini_integration
        self.visual_extractor = visual_context_extractor
        self.config = config
        
        # Reformulation templates
        self.reformulation_templates = self._load_reformulation_templates()
        
        logger.info("Query Reformulator initialized")
    
    def _load_reformulation_templates(self) -> Dict:
        """Load reformulation templates for different question types"""
        return {
            'descriptive': {
                'pattern': ['what', 'describe', 'identify', 'show'],
                'template': """
                Original question: "{original_question}"
                Image context: {visual_context}
                Anatomical context: {anatomical_context}
                
                Reformulate this question to be specific to the visible image content. 
                Make it self-contained by incorporating relevant visual details.
                Focus on what can actually be observed in this specific image.
                
                Guidelines:
                - Reference specific visual features visible in the image
                - Include anatomical location if identifiable
                - Maintain medical accuracy and terminology
                - Make the question answerable from the image alone
                
                Reformulated question:
                """
            },
            'diagnostic': {
                'pattern': ['diagnosis', 'condition', 'disease', 'pathology'],
                'template': """
                Original question: "{original_question}"
                Image context: {visual_context}
                Anatomical context: {anatomical_context}
                
                Reformulate this diagnostic question to be grounded in the specific image findings.
                Reference visible pathological features and their characteristics.
                
                Guidelines:
                - Specify the visible abnormalities or findings
                - Reference anatomical structures shown
                - Include relevant morphological details
                - Frame as analysis of visible features
                
                Reformulated question:
                """
            },
            'presence': {
                'pattern': ['present', 'visible', 'seen', 'shown', 'evidence'],
                'template': """
                Original question: "{original_question}"
                Image context: {visual_context}
                Anatomical context: {anatomical_context}
                
                Reformulate this presence-based question to specifically reference 
                the visible image content and anatomical structures.
                
                Guidelines:
                - Specify exactly what structures/regions to examine
                - Reference image characteristics and quality
                - Include relevant anatomical landmarks
                - Make the evaluation criteria explicit
                
                Reformulated question:
                """
            },
            'comparison': {
                'pattern': ['compare', 'difference', 'similar', 'versus'],
                'template': """
                Original question: "{original_question}"
                Image context: {visual_context}
                Anatomical context: {anatomical_context}
                
                Reformulate this comparison question to focus on the specific 
                features and characteristics visible in this image.
                
                Guidelines:
                - Identify the specific features to compare
                - Reference visible anatomical structures
                - Include morphological characteristics
                - Ground comparison in observable details
                
                Reformulated question:
                """
            },
            'general': {
                'pattern': [],  # Fallback for unmatched questions
                'template': """
                Original question: "{original_question}"
                Image context: {visual_context}
                Anatomical context: {anatomical_context}
                
                Transform this clinical question into a self-contained formulation 
                that is specifically grounded in the visible image content.
                
                Guidelines:
                - Incorporate specific visual details from the image
                - Reference anatomical structures and their appearance
                - Make the question answerable from image observation alone
                - Maintain clinical relevance and accuracy
                
                Reformulated question:
                """
            }
        }
    
    def identify_question_type(self, question: str) -> str:
        """
        Identify the type of clinical question to select appropriate template
        
        Args:
            question: Original question text
            
        Returns:
            Question type string
        """
        question_lower = question.lower()
        
        for question_type, template_info in self.reformulation_templates.items():
            if question_type == 'general':
                continue
                
            patterns = template_info['pattern']
            if any(pattern in question_lower for pattern in patterns):
                logger.debug(f"Identified question type: {question_type}")
                return question_type
        
        logger.debug("Using general question type")
        return 'general'
    
    def reformulate_question(self, image: Image.Image, question: str) -> Dict:
        """
        Reformulate a clinical question to be grounded in specific image content
        
        Args:
            image: PIL Image
            question: Original clinical question
            
        Returns:
            Dictionary containing reformulation results
        """
        logger.info(f"Reformulating question: {question}")
        
        try:
            # Extract visual context
            visual_context = self.visual_extractor.extract_complete_context(image, question)
            
            # Identify question type
            question_type = self.identify_question_type(question)
            
            # Get appropriate template
            template = self.reformulation_templates[question_type]['template']
            
            # Format template with context
            formatted_prompt = template.format(
                original_question=question,
                visual_context=visual_context['visual_description'],
                anatomical_context=visual_context['anatomical_context']
            )
            
            # Generate reformulated question using Gemini
            reformulated = self._generate_reformulated_question(formatted_prompt)
            
            # Post-process and validate
            validated_question = self._validate_reformulated_question(reformulated, question)
            
            # Compile results
            result = {
                'original_question': question,
                'reformulated_question': validated_question,
                'question_type': question_type,
                'visual_context': visual_context,
                'reformulation_quality': self._assess_reformulation_quality(question, validated_question),
                'success': True
            }
            
            logger.info(f"Successfully reformulated question")
            logger.debug(f"Reformulated: {validated_question}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error reformulating question: {e}")
            
            # Return fallback result
            return {
                'original_question': question,
                'reformulated_question': question,  # Fallback to original
                'question_type': 'unknown',
                'visual_context': {},
                'reformulation_quality': {'score': 0.0, 'issues': ['reformulation_failed']},
                'success': False,
                'error': str(e)
            }
    
    def _generate_reformulated_question(self, prompt: str) -> str:
        """Generate reformulated question using Gemini"""
        try:
            # Use Gemini to generate reformulated question
            response = self.gemini.model.generate_content(
                prompt,
                generation_config=self.gemini.generation_config
            )
            
            reformulated = response.text.strip()
            
            # Extract just the reformulated question if it includes extra text
            if "Reformulated question:" in reformulated:
                reformulated = reformulated.split("Reformulated question:")[-1].strip()
            
            return reformulated
            
        except Exception as e:
            logger.error(f"Error generating reformulated question: {e}")
            raise
    
    def _validate_reformulated_question(self, reformulated: str, original: str) -> str:
        """
        Validate and clean up the reformulated question
        
        Args:
            reformulated: Generated reformulated question
            original: Original question
            
        Returns:
            Validated reformulated question
        """
        # Basic cleanup
        cleaned = reformulated.strip()
        
        # Remove any quotes
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        
        # Ensure it ends with question mark
        if not cleaned.endswith('?'):
            cleaned += '?'
        
        # Ensure it's not empty or too similar to original
        if len(cleaned) < 10 or cleaned.lower() == original.lower():
            logger.warning("Reformulated question is too short or identical to original")
            return original
        
        return cleaned
    
    def _assess_reformulation_quality(self, original: str, reformulated: str) -> Dict:
        """
        Assess the quality of question reformulation
        
        Args:
            original: Original question
            reformulated: Reformulated question
            
        Returns:
            Quality assessment dictionary
        """
        quality_score = 0.0
        issues = []
        
        # Length check
        if len(reformulated) > len(original):
            quality_score += 0.3  # Longer is often more specific
        else:
            issues.append('reformulated_shorter')
        
        # Medical terminology check
        medical_terms = ['anatomical', 'pathological', 'clinical', 'morphological', 
                        'tissue', 'lesion', 'mass', 'structure', 'organ']
        if any(term in reformulated.lower() for term in medical_terms):
            quality_score += 0.2
        
        # Image grounding check
        image_terms = ['visible', 'shown', 'depicted', 'image', 'displayed', 
                      'observed', 'present in', 'seen in']
        if any(term in reformulated.lower() for term in image_terms):
            quality_score += 0.3
        else:
            issues.append('lacks_image_grounding')
        
        # Specificity check
        specific_terms = ['specific', 'particular', 'characteristic', 'distinctive',
                         'location', 'region', 'area', 'zone']
        if any(term in reformulated.lower() for term in specific_terms):
            quality_score += 0.2
        
        return {
            'score': min(quality_score, 1.0),
            'issues': issues,
            'length_ratio': len(reformulated) / len(original),
            'has_medical_terms': any(term in reformulated.lower() for term in medical_terms),
            'has_image_grounding': any(term in reformulated.lower() for term in image_terms)
        }
    
    def batch_reformulate(self, image_question_pairs: List[Tuple[Image.Image, str]]) -> List[Dict]:
        """
        Reformulate multiple questions in batch
        
        Args:
            image_question_pairs: List of (image, question) tuples
            
        Returns:
            List of reformulation results
        """
        logger.info(f"Batch reformulating {len(image_question_pairs)} questions")
        
        results = []
        for i, (image, question) in enumerate(image_question_pairs):
            logger.info(f"Processing {i+1}/{len(image_question_pairs)}")
            result = self.reformulate_question(image, question)
            results.append(result)
        
        # Compute batch statistics
        successful = sum(1 for r in results if r['success'])
        avg_quality = sum(r['reformulation_quality']['score'] for r in results) / len(results)
        
        logger.info(f"Batch reformulation complete: {successful}/{len(results)} successful, "
                   f"average quality: {avg_quality:.3f}")
        
        return results
    
    def save_reformulation_results(self, results: List[Dict], output_path: str):
        """
        Save reformulation results to file
        
        Args:
            results: List of reformulation results
            output_path: Output file path
        """
        try:
            # Prepare data for saving (remove non-serializable objects)
            serializable_results = []
            for result in results:
                clean_result = {
                    'original_question': result['original_question'],
                    'reformulated_question': result['reformulated_question'],
                    'question_type': result['question_type'],
                    'reformulation_quality': result['reformulation_quality'],
                    'success': result['success']
                }
                
                # Add visual context summary (exclude raw tensors)
                if 'visual_context' in result and result['visual_context']:
                    clean_result['visual_description'] = result['visual_context'].get('visual_description', '')
                    clean_result['anatomical_context'] = result['visual_context'].get('anatomical_context', '')
                
                serializable_results.append(clean_result)
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reformulation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving reformulation results: {e}")
