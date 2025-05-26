#!/usr/bin/env python
"""
🎯 SIMPLE BASELINE COMPARISON: So sánh với BLIP-only baseline
Day 2: Generate baseline comparison for paper
"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

class SimpleBaselineComparison:
    def __init__(self, config_path="configs/config.yaml", model_path="checkpoints/blip/checkpoints/best_hf_model"):
        self.config = Config(config_path)
        self.logger = setup_logger('baseline_comparison', self.config['logging']['save_dir'])
        
        # Load BLIP model
        self.blip_model = self.load_blip_model(model_path)
        
        # Load semantic model if available
        if SEMANTIC_AVAILABLE:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.semantic_model = None
            
    def load_blip_model(self, model_path):
        """Load BLIP model for baseline comparison"""
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
    
    def compute_semantic_similarity(self, predicted, ground_truth):
        """Compute semantic similarity"""
        if not SEMANTIC_AVAILABLE or self.semantic_model is None:
            # Simple word overlap fallback
            pred_words = set(predicted.lower().split())
            gt_words = set(ground_truth.lower().split())
            
            if len(gt_words) == 0:
                return 0.0
            
            overlap = len(pred_words.intersection(gt_words))
            return overlap / len(gt_words)
        
        try:
            pred_emb = self.semantic_model.encode([predicted])
            gt_emb = self.semantic_model.encode([ground_truth])
            
            similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
            return float(similarity)
            
        except Exception:
            return 0.0
    
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
        
        # Take first num_samples for consistency
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
    
    def run_blip_only_baseline(self, test_samples, output_dir="data/baseline_results"):
        """Run BLIP-only baseline (no enhancements)"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"🔬 Running BLIP-only baseline on {len(test_samples)} samples...")
        
        baseline_results = []
        successful_samples = 0
        
        for i, sample in enumerate(test_samples):
            try:
                self.logger.info(f"Processing sample {i+1}/{len(test_samples)}: {sample['image_id']}")
                
                # Load image
                image = Image.open(sample['image_path']).convert('RGB')
                
                # BLIP prediction only (no enhancements)
                blip_answer = self.blip_model.predict(image, sample['question'])
                
                # Compute similarity
                semantic_score = self.compute_semantic_similarity(blip_answer, sample['answer'])
                
                result = {
                    'sample_id': sample['image_id'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'blip_only_answer': blip_answer,
                    'semantic_similarity': semantic_score,
                    'success': True
                }
                
                baseline_results.append(result)
                successful_samples += 1
                
                self.logger.info(f"✅ BLIP answer: {blip_answer[:50]}...")
                self.logger.info(f"📊 Semantic similarity: {semantic_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {sample['image_id']}: {e}")
                
                error_result = {
                    'sample_id': sample['image_id'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'blip_only_answer': f"Error: {str(e)}",
                    'semantic_similarity': 0.0,
                    'success': False
                }
                baseline_results.append(error_result)
                continue
        
        # Save results
        results_file = os.path.join(output_dir, "blip_only_baseline_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        
        # Compute summary statistics
        semantic_scores = [r['semantic_similarity'] for r in baseline_results if r['success']]
        
        summary = {
            'total_samples': len(test_samples),
            'successful_samples': successful_samples,
            'success_rate': successful_samples / len(test_samples),
            'avg_semantic_similarity': np.mean(semantic_scores) if semantic_scores else 0,
            'std_semantic_similarity': np.std(semantic_scores) if semantic_scores else 0,
            'median_semantic_similarity': np.median(semantic_scores) if semantic_scores else 0
        }
        
        summary_file = os.path.join(output_dir, "blip_only_baseline_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\n📊 BLIP-ONLY BASELINE SUMMARY:")
        self.logger.info(f"Success Rate: {summary['success_rate']*100:.1f}%")
        self.logger.info(f"Avg Semantic Similarity: {summary['avg_semantic_similarity']:.3f} ± {summary['std_semantic_similarity']:.3f}")
        self.logger.info(f"Results saved to: {output_dir}")
        
        return baseline_results, summary

def main():
    print("🎯 Running BLIP-only Baseline Comparison")
    print("="*40)
    
    # Initialize comparison
    baseline_comp = SimpleBaselineComparison()
    
    if baseline_comp.blip_model is None:
        print("❌ Failed to load BLIP model. Exiting.")
        return
    
    # Load test samples 
    test_samples = baseline_comp.load_test_samples(num_samples=50)
    print(f"📊 Loaded {len(test_samples)} test samples")
    
    # Run baseline
    baseline_results, summary = baseline_comp.run_blip_only_baseline(test_samples)
    
    print(f"\n✅ Baseline comparison completed!")
    print(f"📈 BLIP-only performance: {summary['avg_semantic_similarity']:.3f} semantic similarity")

if __name__ == "__main__":
    main()
