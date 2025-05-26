#!/usr/bin/env python
"""
🎯 BLIP-only Baseline Comparison - Torch compatibility fixed
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger

# 🔧 TORCH COMPATIBILITY FIX
import torch
if not hasattr(torch, 'get_default_device'):
    def _get_default_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.get_default_device = _get_default_device

from src.models.blip2.model import BLIP2VQA

class BaselineComparison:
    def __init__(self, config_path="configs/config.yaml", model_path="checkpoints/blip/checkpoints/best_hf_model"):
        self.config = Config(config_path)
        self.logger = setup_logger('baseline_comparison', self.config['logging']['save_dir'])
        
        # Load BLIP model
        self.blip_model = self.load_blip_model(model_path)
        
    def load_blip_model(self, model_path):
        """Load BLIP model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            model = BLIP2VQA(self.config, train_mode=False)
            model.device = device
            
            if os.path.isdir(model_path):
                model.model = type(model.model).from_pretrained(model_path)
                model.model.to(device)
                self.logger.info("Loaded BLIP model from HuggingFace directory")
            else:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.model.load_state_dict(checkpoint)
            
            model.model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading BLIP model: {e}")
            return None
    
    def compute_similarity(self, predicted, ground_truth):
        """Compute similarity score"""
        if not predicted.strip() or not ground_truth.strip():
            return 0.0
        
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()
        
        # Exact match
        if pred_lower == gt_lower:
            return 1.0
        
        # Substring containment
        if gt_lower in pred_lower or pred_lower in gt_lower:
            return 0.8
        
        # Word overlap (Jaccard similarity)
        pred_words = set(pred_lower.split())
        gt_words = set(gt_lower.split())
        
        intersection = len(pred_words.intersection(gt_words))
        union = len(pred_words.union(gt_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def load_test_samples(self, num_samples=50):
        """Load test samples"""
        test_questions_file = self.config['data']['test_questions']
        test_images_dir = self.config['data']['test_images']
        
        questions = []
        with open(test_questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    questions.append(item)
                except:
                    continue
        
        selected_questions = questions[:num_samples]
        
        samples = []
        for item in selected_questions:
            image_id = item['image_id']
            
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = Path(test_images_dir) / f"{image_id}{ext}"
                if img_path.exists():
                    samples.append({
                        'image_id': image_id,
                        'question': item['question'],
                        'answer': item['answer'],
                        'image_path': str(img_path)
                    })
                    break
        
        return samples
    
    def run_baseline(self, test_samples, output_dir="data/baseline_results"):
        """Run BLIP-only baseline"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"🔬 Running BLIP-only baseline on {len(test_samples)} samples...")
        
        baseline_results = []
        successful_samples = 0
        
        for i, sample in enumerate(test_samples):
            try:
                self.logger.info(f"Processing sample {i+1}/{len(test_samples)}: {sample['image_id']}")
                
                # Load image
                image = Image.open(sample['image_path']).convert('RGB')
                
                # BLIP prediction only
                blip_answer = self.blip_model.predict(image, sample['question'])
                
                # Compute similarity
                similarity_score = self.compute_similarity(blip_answer, sample['answer'])
                
                result = {
                    'sample_id': sample['image_id'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'blip_only_answer': blip_answer,
                    'similarity_score': similarity_score,
                    'success': True
                }
                
                baseline_results.append(result)
                successful_samples += 1
                
                self.logger.info(f"✅ BLIP answer: {blip_answer[:50]}...")
                self.logger.info(f"📊 Similarity: {similarity_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {sample['image_id']}: {e}")
                
                error_result = {
                    'sample_id': sample['image_id'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'blip_only_answer': f"Error: {str(e)}",
                    'similarity_score': 0.0,
                    'success': False
                }
                baseline_results.append(error_result)
                continue
        
        # Save results
        results_file = os.path.join(output_dir, "blip_baseline_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        
        # Compute summary
        similarity_scores = [r['similarity_score'] for r in baseline_results if r['success']]
        
        summary = {
            'total_samples': len(test_samples),
            'successful_samples': successful_samples,
            'success_rate': successful_samples / len(test_samples),
            'avg_similarity': np.mean(similarity_scores) if similarity_scores else 0,
            'std_similarity': np.std(similarity_scores) if similarity_scores else 0,
            'median_similarity': np.median(similarity_scores) if similarity_scores else 0
        }
        
        summary_file = os.path.join(output_dir, "blip_baseline_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\n📊 BLIP-ONLY BASELINE SUMMARY:")
        self.logger.info(f"Success Rate: {summary['success_rate']*100:.1f}%")
        self.logger.info(f"Avg Similarity: {summary['avg_similarity']:.3f} ± {summary['std_similarity']:.3f}")
        self.logger.info(f"Results saved to: {output_dir}")
        
        return baseline_results, summary

def main():
    print("🎯 Running BLIP-only Baseline Comparison")
    print("="*40)
    
    # Initialize comparison
    baseline_comp = BaselineComparison()
    
    if baseline_comp.blip_model is None:
        print("❌ Failed to load BLIP model. Exiting.")
        return
    
    # Load test samples (same number as evaluation)
    test_samples = baseline_comp.load_test_samples(num_samples=50)
    print(f"📊 Loaded {len(test_samples)} test samples")
    
    # Run baseline
    baseline_results, summary = baseline_comp.run_baseline(test_samples)
    
    print(f"\n✅ Baseline comparison completed!")
    print(f"📈 BLIP-only performance: {summary['avg_similarity']:.3f} similarity score")

if __name__ == "__main__":
    main()
