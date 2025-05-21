#!/usr/bin/env python
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.data_loader import get_data_loader

def denormalize_image(image, mean, std):
    """Khôi phục hình ảnh từ tensor đã normalize"""
    image = image.clone().detach()
    # Đảo ngược quá trình normalize
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    # Chuyển về khoảng [0, 1]
    return torch.clamp(image, 0, 1)

def visualize_batch(batch, config, num_samples=3):
    """Hiển thị một số mẫu từ batch"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))
    
    # Thông số normalize
    mean = config['preprocessing']['image']['normalize']['mean']
    std = config['preprocessing']['image']['normalize']['std']
    
    # Chọn số mẫu để hiển thị
    indices = np.random.choice(len(batch['image']), min(num_samples, len(batch['image'])), replace=False)
    
    for i, idx in enumerate(indices):
        # Lấy và denormalize hình ảnh
        image = batch['image'][idx].cpu()
        image = denormalize_image(image, mean, std)
        
        # Chuyển về định dạng phù hợp để hiển thị
        image = image.permute(1, 2, 0).numpy()
        
        # Hiển thị hình ảnh và thông tin
        if num_samples > 1:
            ax = axes[i]
        else:
            ax = axes
        
        ax.imshow(image)
        ax.set_title(f"ID: {batch['image_id'][idx]}")
        ax.set_xlabel(f"Q: {batch['question'][idx][:50]}...")
        ax.set_ylabel(f"A: {batch['answer'][idx][:50]}...")
        ax.axis('on')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Test DataLoader')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Data split to test')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='data/tests',
                        help='Directory to save visualization')
    args = parser.parse_args()
    
    print(f"Testing DataLoader for {args.split} split...")
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('dataloader_test', config['logging']['save_dir'], level='INFO')
    
    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo DataLoader
    dataloader, dataset = get_data_loader(
        config, 
        split=args.split, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    logger.info(f"Loaded {args.split} dataset with {len(dataset)} samples")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    # Thử lấy một batch
    try:
        iterator = iter(dataloader)
        batch = next(iterator)
        logger.info("Successfully loaded a batch")
        
        # Hiển thị thông tin batch
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Image tensor shape: {batch['image'].shape}")
        logger.info(f"Sample questions: {batch['question'][0][:50]}...")
        
        # Visualize batch
        fig = visualize_batch(batch, config)
        output_path = os.path.join(args.output_dir, f"{args.split}_batch_samples.png")
        fig.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading batch: {e}")
        return False

if __name__ == "__main__":
    main()
