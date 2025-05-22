# Add debug validation method to ChainOfThoughtGenerator
def _validate_reasoning_chain_debug(self, reasoning_chain: Dict) -> Dict:
    """Debug version of validation"""
    logger.debug("=== VALIDATION DEBUG ===")
    
    chain_data = reasoning_chain.get('reasoning_chain', {})
    logger.debug(f"Chain data keys: {chain_data.keys()}")
    
    # Use templates validation
    template_validation = self.templates.validate_reasoning_chain(chain_data)
    logger.debug(f"Template validation: {template_validation}")
    
    # Add medical knowledge validation
    steps = chain_data.get('steps', [])
    medical_validation = self.knowledge_base.validate_clinical_reasoning(steps)
    logger.debug(f"Medical validation: {medical_validation}")
    
    # IMPROVED: Confidence-aware validation
    overall_confidence = chain_data.get('overall_confidence', 0.0)
    confidence_validity = overall_confidence >= 0.5
    
    # DEBUG: Check individual scores
    template_completeness = template_validation.get('completeness_score', 0)
    template_consistency = template_validation.get('consistency_score', 0)
    medical_accuracy = medical_validation.get('medical_accuracy_score', 0)
    medical_consistency = medical_validation.get('logical_consistency_score', 0)
    
    logger.debug(f"Individual scores: completeness={template_completeness}, consistency={template_consistency}, medical_acc={medical_accuracy}, medical_cons={medical_consistency}, confidence={overall_confidence}")
    
    # Calculate combined score
    individual_scores = [template_completeness, template_consistency, medical_accuracy, medical_consistency, overall_confidence]
    valid_scores = [score for score in individual_scores if score is not None and score > 0]
    
    if valid_scores:
        combined_score = sum(valid_scores) / len(valid_scores)
    else:
        combined_score = overall_confidence  # Fallback to confidence only
    
    logger.debug(f"Combined score calculation: {valid_scores} -> {combined_score}")
    
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
        'combined_score': combined_score,
        'debug_info': {
            'individual_scores': individual_scores,
            'valid_scores': valid_scores,
            'calculation_method': 'average_of_valid_scores'
        }
    }
    
    logger.debug(f"Final validation result: {combined_validation}")
    return combined_validation
