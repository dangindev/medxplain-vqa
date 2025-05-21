#!/usr/bin/env python
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import platform
from PIL import Image
import google.generativeai as genai
import json
import time

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config, load_api_keys
from src.utils.logger import setup_logger

def check_gpu():
    print("=== GPU Information ===")
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Kiểm tra bộ nhớ GPU
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Thử tensor operation trên GPU
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Matrix multiplication time on GPU: {(end - start) * 1000:.2f} ms")
    else:
        print("GPU is not available")

def check_libraries():
    print("\n=== Library Information ===")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        import torchvision
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision version: Not installed")

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers version: Not installed")

    print(f"Pillow version: {Image.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"Generative AI version: {genai.__version__ if 'genai' in globals() else 'Not installed'}")

def test_gemini_api(api_key):
    print("\n=== Testing Gemini API ===")
    if not api_key:
        print("No Gemini API key provided.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-1.5-pro')
        response = model.generate_content("Hello! Please respond with a brief greeting.")
        print(f"Gemini response: {response.text}")
        return True
    except Exception as e:
        print(f"Error testing Gemini API: {e}")
        return False

def test_data_loading(config):
    print("\n=== Testing Data Loading ===")
    try:
        # Kiểm tra tồn tại của các đường dẫn dữ liệu
        for split in ['train', 'val', 'test']:
            image_dir = config['data'][f'{split}_images']
            questions_file = config['data'][f'{split}_questions']
            
            print(f"{split} images directory: {os.path.exists(image_dir)}")
            print(f"{split} questions file: {os.path.exists(questions_file)}")
            
            # Thử tải một số hình ảnh
            if os.path.exists(image_dir):
                image_files = list(Path(image_dir).glob('*.*'))[:3]
                for img_file in image_files:
                    try:
                        img = Image.open(img_file)
                        print(f"Successfully loaded image: {img_file} (Size: {img.size})")
                    except Exception as e:
                        print(f"Error loading image {img_file}: {e}")
            
            # Thử tải dữ liệu câu hỏi
            if os.path.exists(questions_file):
                with open(questions_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        try:
                            qa = json.loads(line)
                            print(f"Sample Q&A: {qa['image_id']} - Q: {qa['question'][:50]}... A: {qa['answer'][:50]}...")
                        except Exception as e:
                            print(f"Error parsing line {i}: {e}")
        
        return True
    except Exception as e:
        print(f"Error testing data loading: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test environment and data')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    parser.add_argument('--api-keys', type=str, default='configs/api_keys.yaml',
                        help='Path to the API keys file')
    args = parser.parse_args()
    
    print("Starting environment test...")
    
    # Load config
    try:
        config = Config(args.config)
        print(f"Successfully loaded config from {args.config}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Load API keys
    try:
        api_keys = load_api_keys(args.api_keys)
        print(f"Successfully loaded API keys from {args.api_keys}")
    except Exception as e:
        print(f"Error loading API keys: {e}")
        api_keys = {}
    
    # Kiểm tra GPU
    check_gpu()
    
    # Kiểm tra thư viện
    check_libraries()
    
    # Kiểm tra Gemini API
    gemini_api_key = api_keys.get('gemini', {}).get('api_key', '')
    test_gemini_api(gemini_api_key)
    
    # Kiểm tra tải dữ liệu
    test_data_loading(config)
    
    print("\nEnvironment test completed!")

if __name__ == "__main__":
    main()
