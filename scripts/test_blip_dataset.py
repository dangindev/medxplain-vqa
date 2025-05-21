#!/usr/bin/env python
import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA
from src.models.blip2.dataset import PathVQAFineTuneDataset, get_dataloader

def denormalize_image(image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    """Chuyển tensor về hình ảnh"""
    # Sao chép để không làm thay đổi tensor gốc
    image = image.clone().detach()
    
    # Đảo ngược quá trình normalize
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    
    # Chuyển về khoảng [0, 1]
    return torch.clamp(image, 0, 1).permute(1, 2, 0).cpu().numpy()

def visualize_batch(batch, processor, num_samples=3, output_path=None):
    """Hiển thị một số mẫu từ batch"""
    if num_samples > len(batch['pixel_values']):
        num_samples = len(batch['pixel_values'])
    
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 5 * num_samples))
    
    if num_samples == 1:
        axs = [axs]
    
    for i in range(num_samples):
        # Lấy thông tin
        img = batch['pixel_values'][i]
        question = batch['question_text'][i]
        answer = batch['answer_text'][i]
        
        # Chuyển tensor về hình ảnh
        img_np = denormalize_image(img)
        
        # Hiển thị
        axs[i].imshow(img_np)
        axs[i].set_title(f"Q: {question}\nA: {answer}", fontsize=12)
        axs[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.close()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Test PathVQA Dataset for Fine-tuning')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                      help='Data split to visualize')
    parser.add_argument('--num-samples', type=int, default=3,
                      help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='data/tests',
                      help='Directory to save visualization')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('test_dataset', config['logging']['save_dir'], level='INFO')
    logger.info(f"Testing PathVQA Dataset for {args.split} split")
    
    # Tạo thư mục đầu ra
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Tải model để lấy processor
    logger.info("Loading BLIP model to get processor...")
    model = BLIP2VQA(config, train_mode=False)
    processor = model.processor
    
    # Tạo dataset
    logger.info(f"Creating {args.split} dataset...")
    dataset = PathVQAFineTuneDataset(config, processor, split=args.split)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Lấy một số mẫu từ dataset
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    logger.info(f"Selected indices: {indices}")
    
    # Tạo batch
    batch = {'pixel_values': [], 'input_ids': [], 'attention_mask': [], 'labels': [], 
            'question_text': [], 'answer_text': [], 'image_id': []}
    
    for idx in indices:
        sample = dataset[idx]
        batch['pixel_values'].append(sample['pixel_values'])
        batch['input_ids'].append(sample['input_ids'])
        batch['attention_mask'].append(sample['attention_mask'])
        batch['labels'].append(sample['labels'])
        batch['question_text'].append(sample['question_text'])
        batch['answer_text'].append(sample['answer_text'])
        batch['image_id'].append(sample['image_id'])
    
    # Chuyển sang tensor
    batch['pixel_values'] = torch.stack(batch['pixel_values'])
    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['attention_mask'] = torch.stack(batch['attention_mask'])
    batch['labels'] = torch.stack(batch['labels'])
    
    # Hiển thị mẫu
    logger.info("Visualizing samples...")
    output_path = output_dir / f"pathvqa_{args.split}_samples.png"
    visualize_batch(batch, processor, num_samples=args.num_samples, output_path=output_path)
    
    # Kiểm tra dataloader
    logger.info("Testing dataloader...")
    dataloader, _ = get_dataloader(config, processor, split=args.split, batch_size=4)
    
    # Lấy một batch từ dataloader
    logger.info("Getting a batch from dataloader...")
    for batch in dataloader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Pixel values shape: {batch['pixel_values'].shape}")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        
        # Hiển thị batch từ dataloader
        output_path = output_dir / f"pathvqa_{args.split}_batch.png"
        visualize_batch(batch, processor, num_samples=min(3, len(batch['pixel_values'])), output_path=output_path)
        
        # Chi hiển thị một batch
        break
    
    logger.info("Dataset and dataloader test completed successfully")

if __name__ == "__main__":
    main()