#!/usr/bin/env python
import os
import sys
import torch
import argparse
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

def load_model(config, model_path, logger):
    """Tải mô hình đã huấn luyện"""
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Xác định thiết bị
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Khởi tạo mô hình
        model = BLIP2VQA(config, train_mode=False)
        
        if os.path.isdir(model_path):
            # Nếu là thư mục HuggingFace, sử dụng from_pretrained
            model.model = type(model.model).from_pretrained(model_path)
            model.to(device)  # Đảm bảo mô hình ở đúng thiết bị
            logger.info("Loaded model from HuggingFace directory")
        else:
            # Nếu là file checkpoint PyTorch
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                logger.warning("Checkpoint format is not as expected, trying direct load")
                model.model.load_state_dict(checkpoint)
        
        # Đưa mô hình về chế độ evaluation và đúng thiết bị
        model.model.eval()
        model.device = device  # Đảm bảo model biết đang ở thiết bị nào
        logger.info(f"Model loaded successfully on {device}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_test_samples(config, num_samples=5, seed=42):
    """Tải một số mẫu ngẫu nhiên từ tập test"""
    random.seed(seed)
    
    # Đường dẫn file câu hỏi test
    test_questions_file = config['data']['test_questions']
    test_images_dir = config['data']['test_images']
    
    # Đọc file câu hỏi
    questions = []
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                questions.append(item)
            except json.JSONDecodeError:
                continue
    
    # Chọn ngẫu nhiên num_samples mẫu
    selected_samples = random.sample(questions, min(num_samples, len(questions)))
    
    # Tìm đường dẫn hình ảnh
    samples = []
    for item in selected_samples:
        image_id = item['image_id']
        # Thử các phần mở rộng phổ biến
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

def predict_and_visualize(model, samples, output_dir, logger):
    """Thực hiện dự đoán và trực quan hóa kết quả"""
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Vẽ hình ảnh, câu hỏi, câu trả lời mẫu và câu trả lời dự đoán
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 7 * len(samples)))
    
    if len(samples) == 1:
        axes = [axes]
    
    predictions = []
    for i, sample in enumerate(samples):
        # Tải hình ảnh
        image = Image.open(sample['image_path'])
        
        # Dự đoán câu trả lời
        try:
            predicted_answer = model.predict(image, sample['question'])
            
            # Lưu kết quả
            predictions.append({
                'image_id': sample['image_id'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'prediction': predicted_answer
            })
            
            # Vẽ kết quả
            axes[i].imshow(image)
            axes[i].set_title(f"Sample {i+1}: {sample['image_id']}", fontsize=14)
            
            # Thêm văn bản
            axes[i].text(
                0, -0.1, 
                f"Question: {sample['question']}", 
                transform=axes[i].transAxes, 
                fontsize=12
            )
            axes[i].text(
                0, -0.15, 
                f"Ground Truth: {sample['answer']}", 
                transform=axes[i].transAxes, 
                fontsize=12, 
                color='green'
            )
            axes[i].text(
                0, -0.2, 
                f"Prediction: {predicted_answer}", 
                transform=axes[i].transAxes, 
                fontsize=12, 
                color='blue'
            )
            
            # Tắt trục
            axes[i].axis('off')
            
        except Exception as e:
            logger.error(f"Error predicting for sample {i+1}: {e}")
    
    # Lưu hình ảnh
    plt.tight_layout()
    output_path = os.path.join(output_dir, "test_samples_predictions.png")
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")
    
    # Lưu kết quả dự đoán
    predictions_path = os.path.join(output_dir, "test_predictions.json")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {predictions_path}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned BLIP model on specific examples')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to model checkpoint or directory')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='data/inference_results', 
                      help='Directory to save results')
    parser.add_argument('--device', type=str, default=None, 
                      help='Device to use (cuda or cpu). If None, will use CUDA if available.')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('test_inference', config['logging']['save_dir'], level='INFO')
    logger.info("Starting inference test")
    
    # Tải mô hình
    model = load_model(config, args.model_path, logger)
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Tải mẫu test
    logger.info(f"Loading {args.num_samples} test samples")
    samples = load_test_samples(config, args.num_samples)
    
    if not samples:
        logger.error("No valid test samples found. Exiting.")
        return
    
    logger.info(f"Loaded {len(samples)} test samples")
    
    # Dự đoán và trực quan hóa
    predictions = predict_and_visualize(model, samples, args.output_dir, logger)
    
    # Hiển thị kết quả
    for i, pred in enumerate(predictions):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Question: {pred['question']}")
        logger.info(f"  Ground truth: {pred['ground_truth']}")
        logger.info(f"  Prediction: {pred['prediction']}")
    
    logger.info("Inference test completed")

if __name__ == "__main__":
    main()
