#!/usr/bin/env python
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.text_processor import TextProcessor

def preprocess_split(config, split, image_processor, text_processor, logger):
    """Tiền xử lý một tập dữ liệu (train/val/test)"""
    logger.info(f"Processing {split} split...")
    
    # Đường dẫn dữ liệu
    image_dir = config['data'][f'{split}_images']
    questions_file = config['data'][f'{split}_questions']
    
    # Tạo thư mục lưu dữ liệu đã xử lý
    processed_dir = Path(config['data']['processed_dir'])
    processed_dir.mkdir(exist_ok=True, parents=True)
    processed_splits_dir = processed_dir / split
    processed_splits_dir.mkdir(exist_ok=True)
    
    # Xử lý hình ảnh
    logger.info(f"Processing {split} images...")
    try:
        processed_images = image_processor.preprocess_dataset(image_dir, split)
        logger.info(f"Processed {len(processed_images)} {split} images")
    except Exception as e:
        logger.error(f"Error processing {split} images: {e}")
        logger.error(traceback.format_exc())
        processed_images = {}
    
    # Xử lý text
    logger.info(f"Processing {split} questions and answers...")
    try:
        processed_text = text_processor.process_dataset(questions_file, split)
        logger.info(f"Processed {len(processed_text)} {split} question-answer pairs")
    except Exception as e:
        logger.error(f"Error processing {split} questions: {e}")
        logger.error(traceback.format_exc())
        processed_text = []
    
    # Tạo file mapping để dễ dàng truy cập
    mapping = {
        'split': split,
        'questions_count': len(processed_text),
        'images_count': len(processed_images),
        'processed_questions_file': str(Path(text_processor.processed_dir) / f"{split}_processed.jsonl"),
        'processed_images_dir': str(Path(image_processor.processed_dir) / split)
    }
    
    with open(processed_dir / f"{split}_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"Completed processing {split} split")
    return mapping

def main():
    parser = argparse.ArgumentParser(description='Preprocess PathVQA dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to the config file')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Setup logger
    logger = setup_logger('data_preprocessing', config['logging']['save_dir'], level='INFO')
    logger.info("Starting PathVQA dataset preprocessing")
    
    try:
        # Khởi tạo processors
        logger.info("Initializing image processor...")
        image_processor = ImageProcessor(config)
        
        logger.info("Initializing text processor...")
        text_processor = TextProcessor(config)
        
        # Xử lý từng tập dữ liệu
        train_mapping = preprocess_split(config, 'train', image_processor, text_processor, logger)
        val_mapping = preprocess_split(config, 'val', image_processor, text_processor, logger)
        test_mapping = preprocess_split(config, 'test', image_processor, text_processor, logger)
        
        # Lưu tổng hợp mapping
        mapping_summary = {
            'train': train_mapping,
            'val': val_mapping,
            'test': test_mapping,
            'processed_root': str(Path(config['data']['processed_dir']))
        }
        
        with open(Path(config['data']['processed_dir']) / "dataset_mapping.json", 'w') as f:
            json.dump(mapping_summary, f, indent=2)
        
        logger.info("Preprocessing complete!")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
