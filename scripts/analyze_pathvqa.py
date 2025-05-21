#!/usr/bin/env python
import os
import json
import argparse
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import Config
from src.utils.logger import setup_logger

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_questions(questions_data):
    # Phân tích câu hỏi
    question_lengths = [len(item['question'].split()) for item in questions_data]
    answer_lengths = [len(item['answer'].split()) for item in questions_data]
    
    # Loại câu hỏi (sử dụng từ đầu tiên)
    question_types = [item['question'].split()[0].lower() for item in questions_data]
    
    # Lấy mẫu một số câu hỏi và câu trả lời
    samples = np.random.choice(questions_data, min(5, len(questions_data)), replace=False)
    
    return {
        'count': len(questions_data),
        'question_lengths': question_lengths,
        'answer_lengths': answer_lengths,
        'question_types': question_types,
        'samples': samples
    }

def analyze_images(images_dir):
    # Lấy thông tin về kích thước hình ảnh
    image_sizes = []
    image_formats = Counter()
    
    image_files = list(Path(images_dir).glob('*.*'))
    for img_path in tqdm(image_files[:min(500, len(image_files))], desc="Analyzing images"):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                image_sizes.append((width, height))
                image_formats[img.format] += 1
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    return {
        'count': len(image_files),
        'sizes': image_sizes,
        'formats': image_formats
    }

def plot_statistics(train_stats, val_stats, test_stats, output_dir):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu cho biểu đồ
    datasets = ['Train', 'Validation', 'Test']
    counts = [train_stats['questions']['count'], val_stats['questions']['count'], test_stats['questions']['count']]
    
    # Biểu đồ số lượng mẫu trong các tập dữ liệu
    plt.figure(figsize=(10, 6))
    sns.barplot(x=datasets, y=counts)
    plt.title('Sample Count per Dataset Split')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_counts.png'))
    plt.close()
    
    # Biểu đồ phân phối độ dài câu hỏi
    plt.figure(figsize=(12, 6))
    sns.histplot(train_stats['questions']['question_lengths'], color='blue', label='Train', alpha=0.7, kde=True)
    sns.histplot(val_stats['questions']['question_lengths'], color='green', label='Val', alpha=0.7, kde=True)
    sns.histplot(test_stats['questions']['question_lengths'], color='red', label='Test', alpha=0.7, kde=True)
    plt.title('Question Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_length_dist.png'))
    plt.close()
    
    # Biểu đồ phân phối độ dài câu trả lời
    plt.figure(figsize=(12, 6))
    sns.histplot(train_stats['questions']['answer_lengths'], color='blue', label='Train', alpha=0.7, kde=True)
    sns.histplot(val_stats['questions']['answer_lengths'], color='green', label='Val', alpha=0.7, kde=True)
    sns.histplot(test_stats['questions']['answer_lengths'], color='red', label='Test', alpha=0.7, kde=True)
    plt.title('Answer Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'answer_length_dist.png'))
    plt.close()
    
    # Biểu đồ loại câu hỏi (top 10)
    all_types = Counter(train_stats['questions']['question_types'] + 
                         val_stats['questions']['question_types'] + 
                         test_stats['questions']['question_types'])
    top_10_types = dict(all_types.most_common(10))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(top_10_types.keys()), y=list(top_10_types.values()))
    plt.title('Top 10 Question Types')
    plt.xlabel('First Word of Question')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_types.png'))
    plt.close()
    
    # Biểu đồ kích thước hình ảnh
    if train_stats['images']['sizes']:
        widths, heights = zip(*train_stats['images']['sizes'])
        plt.figure(figsize=(10, 10))
        plt.scatter(widths, heights, alpha=0.5)
        plt.title('Image Dimensions')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'image_dimensions.png'))
        plt.close()

def save_statistics(train_stats, val_stats, test_stats, output_file):
    stats = {
        'train': {
            'samples': train_stats['questions']['count'],
            'images': train_stats['images']['count'],
            'avg_question_length': np.mean(train_stats['questions']['question_lengths']),
            'avg_answer_length': np.mean(train_stats['questions']['answer_lengths']),
            'question_types': dict(Counter(train_stats['questions']['question_types']).most_common(10))
        },
        'val': {
            'samples': val_stats['questions']['count'],
            'images': val_stats['images']['count'],
            'avg_question_length': np.mean(val_stats['questions']['question_lengths']),
            'avg_answer_length': np.mean(val_stats['questions']['answer_lengths']),
            'question_types': dict(Counter(val_stats['questions']['question_types']).most_common(10))
        },
        'test': {
            'samples': test_stats['questions']['count'],
            'images': test_stats['images']['count'],
            'avg_question_length': np.mean(test_stats['questions']['question_lengths']),
            'avg_answer_length': np.mean(test_stats['questions']['answer_lengths']),
            'question_types': dict(Counter(test_stats['questions']['question_types']).most_common(10))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Analyze PathVQA dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    parser.add_argument('--output-dir', type=str, default='data/analysis',
                        help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('data_analysis', config['logging']['save_dir'], level='INFO')
    logger.info("Starting PathVQA dataset analysis")
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and analyze train data
    logger.info("Analyzing training data...")
    train_questions = load_jsonl(config['data']['train_questions'])
    train_images_stats = analyze_images(config['data']['train_images'])
    train_stats = {
        'questions': analyze_questions(train_questions),
        'images': train_images_stats
    }
    
    # Load and analyze validation data
    logger.info("Analyzing validation data...")
    val_questions = load_jsonl(config['data']['val_questions'])
    val_images_stats = analyze_images(config['data']['val_images'])
    val_stats = {
        'questions': analyze_questions(val_questions),
        'images': val_images_stats
    }
    
    # Load and analyze test data
    logger.info("Analyzing test data...")
    test_questions = load_jsonl(config['data']['test_questions'])
    test_images_stats = analyze_images(config['data']['test_images'])
    test_stats = {
        'questions': analyze_questions(test_questions),
        'images': test_images_stats
    }
    
    # Plot statistics
    logger.info("Generating plots...")
    plot_statistics(train_stats, val_stats, test_stats, args.output_dir)
    
    # Save statistics
    logger.info("Saving statistics...")
    save_statistics(train_stats, val_stats, test_stats, os.path.join(args.output_dir, 'dataset_stats.json'))
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
