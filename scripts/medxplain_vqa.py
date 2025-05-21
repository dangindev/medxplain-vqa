#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import textwrap

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.llm.gemini_integration import GeminiIntegration

def load_model(config, model_path, logger):
    """Tải mô hình BLIP đã trained"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        logger.info(f"Loading BLIP model from {model_path}")
        model = BLIP2VQA(config, train_mode=False)
        model.device = device
        
        if os.path.isdir(model_path):
            model.model = type(model.model).from_pretrained(model_path)
            model.model.to(device)
            logger.info("Loaded model from HuggingFace directory")
        else:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                model.model.load_state_dict(checkpoint)
        
        model.model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading BLIP model: {e}")
        return None

def load_test_samples(config, num_samples=1, random_seed=42):
    """Tải mẫu test ngẫu nhiên"""
    random.seed(random_seed)
    
    # Đường dẫn dữ liệu
    test_questions_file = config['data']['test_questions']
    test_images_dir = config['data']['test_images']
    
    # Tải danh sách câu hỏi
    questions = []
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                questions.append(item)
            except:
                continue
    
    # Chọn ngẫu nhiên
    selected_questions = random.sample(questions, min(num_samples, len(questions)))
    
    # Tìm đường dẫn hình ảnh
    samples = []
    for item in selected_questions:
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

def process_and_visualize(blip_model, gemini, sample, output_dir, logger):
    """Xử lý và trực quan hóa kết quả với bố cục cải tiến"""
    image_path = sample['image_path']
    question = sample['question']
    ground_truth = sample['answer']
    
    # Tải hình ảnh
    image = Image.open(image_path).convert('RGB')
    
    # Dự đoán với BLIP
    logger.info(f"Processing image {sample['image_id']}")
    blip_answer = blip_model.predict(image, question)
    logger.info(f"Initial BLIP answer: {blip_answer}")
    
    # Tạo câu trả lời thống nhất
    logger.info("Generating unified answer...")
    unified_answer = gemini.generate_unified_answer(image, question, blip_answer)
    logger.info(f"Unified answer: {unified_answer}")
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Cải tiến cách hiển thị
    fig = plt.figure(figsize=(10, 12))
    
    # Tạo hai phần: phần trên cho hình ảnh, phần dưới cho văn bản
    ax_image = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_text = plt.subplot2grid((3, 1), (2, 0))
    
    # Hiển thị hình ảnh
    ax_image.imshow(image)
    ax_image.set_title(f"MedXplain-VQA: {sample['image_id']}", fontsize=14)
    ax_image.axis('off')
    
    # Hiển thị văn bản trong phần dưới dưới dạng text box
    text_content = f"Question: {question}\n\nGround truth: {ground_truth}\n\nMedXplain-VQA answer: {unified_answer}"
    ax_text.text(0.01, 0.99, text_content, 
                transform=ax_text.transAxes,
                fontsize=11,
                verticalalignment='top',
                wrap=True)
    ax_text.axis('off')
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu kết quả
    output_file = os.path.join(output_dir, f"medxplain_vqa_{sample['image_id']}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    logger.info(f"Result saved to {output_file}")
    
    # Lưu metadata
    metadata = {
        'image_id': sample['image_id'],
        'image_path': image_path,
        'question': question,
        'ground_truth': ground_truth,
        'blip_answer': blip_answer,
        'unified_answer': unified_answer
    }
    
    metadata_file = os.path.join(output_dir, f"medxplain_vqa_{sample['image_id']}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='MedXplain-VQA with BLIP and Gemini')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, default='checkpoints/blip/checkpoints/best_hf_model', 
                      help='Path to BLIP model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to specific image (optional)')
    parser.add_argument('--question', type=str, default=None, help='Specific question (optional)')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of test samples (if no image specified)')
    parser.add_argument('--output-dir', type=str, default='data/medxplain_results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('medxplain_vqa', config['logging']['save_dir'], level='INFO')
    logger.info("Starting MedXplain-VQA")
    
    # Tải mô hình BLIP
    blip_model = load_model(config, args.model_path, logger)
    if blip_model is None:
        logger.error("Failed to load BLIP model. Exiting.")
        return
    
    # Khởi tạo Gemini
    logger.info("Initializing Gemini")
    try:
        gemini = GeminiIntegration(config)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return
    
    # Xử lý hình ảnh và câu hỏi
    if args.image and args.question:
        # Xử lý hình ảnh và câu hỏi cụ thể
        sample = {
            'image_id': Path(args.image).stem,
            'question': args.question,
            'answer': "Unknown (custom input)",
            'image_path': args.image
        }
        process_and_visualize(blip_model, gemini, sample, args.output_dir, logger)
    else:
        # Tải và xử lý mẫu từ tập test
        logger.info(f"Loading {args.num_samples} test samples")
        samples = load_test_samples(config, args.num_samples)
        
        if not samples:
            logger.error("No test samples found. Exiting.")
            return
        
        logger.info(f"Processing {len(samples)} samples")
        for sample in samples:
            process_and_visualize(blip_model, gemini, sample, args.output_dir, logger)
    
    logger.info("MedXplain-VQA completed")

if __name__ == "__main__":
    main()
