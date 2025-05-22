#!/usr/bin/env python
import os
import sys
import json
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config

def main():
    print("=== Checking Data Structure ===")
    
    # Load config
    config = Config('configs/config.yaml')
    
    # Check question files
    for split in ['train', 'val', 'test']:
        questions_path = config['data'][f'{split}_questions']
        images_dir = config['data'][f'{split}_images']
        
        print(f"\n--- {split.upper()} SPLIT ---")
        print(f"Questions file: {questions_path}")
        print(f"Questions exists: {os.path.exists(questions_path)}")
        
        print(f"Images directory: {images_dir}")
        print(f"Images dir exists: {os.path.exists(images_dir)}")
        
        # Count questions
        if os.path.exists(questions_path):
            with open(questions_path, 'r') as f:
                question_count = sum(1 for line in f)
            print(f"Question count: {question_count}")
            
            # Show sample
            with open(questions_path, 'r') as f:
                sample = json.loads(f.readline())
            print(f"Sample question: {sample}")
        
        # Count images
        if os.path.exists(images_dir):
            image_files = list(Path(images_dir).glob('*.*'))
            print(f"Image count: {len(image_files)}")
            
            if image_files:
                print(f"Sample image: {image_files[0].name}")

if __name__ == "__main__":
    main()
