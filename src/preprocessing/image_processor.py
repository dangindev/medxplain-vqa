import os
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.image_size = tuple(config['preprocessing']['image']['size'])
        
        # Định nghĩa các transforms cho hình ảnh
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['preprocessing']['image']['normalize']['mean'],
                std=config['preprocessing']['image']['normalize']['std']
            )
        ])
        
        # Định nghĩa transform cho việc lưu tiền xử lý
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor(),
        ])
        
        # Tạo thư mục processed nếu chưa tồn tại
        self.processed_dir = Path(config['data']['processed_dir']) / "images"
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def preprocess_image(self, image_path):
        """Tiền xử lý một hình ảnh và trả về tensor"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def preprocess_and_save(self, image_path, save_path=None):
        """Tiền xử lý và lưu hình ảnh"""
        try:
            image = Image.open(image_path).convert('RGB')
            processed = self.preprocess_transform(image)
            
            if save_path is None:
                # Tạo đường dẫn lưu tự động
                image_name = Path(image_path).name
                save_path = self.processed_dir / image_name.replace('.', '_processed.')
            
            # Lưu tensor dưới dạng file .pt
            torch.save(processed, save_path.with_suffix('.pt'))
            return str(save_path)
        except Exception as e:
            print(f"Error processing and saving {image_path}: {e}")
            return None
    
    def preprocess_dataset(self, image_dir, split="train"):
        """Tiền xử lý tất cả hình ảnh trong một thư mục"""
        processed_dir = self.processed_dir / split
        os.makedirs(processed_dir, exist_ok=True)
        
        image_paths = list(Path(image_dir).glob('*.*'))
        processed_paths = {}
        
        for img_path in image_paths:
            image_id = img_path.stem
            save_path = processed_dir / f"{image_id}.pt"
            try:
                self.preprocess_and_save(img_path, save_path)
                processed_paths[image_id] = str(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return processed_paths
    
    def load_processed_image(self, processed_path):
        """Tải hình ảnh đã xử lý từ đường dẫn"""
        try:
            tensor = torch.load(processed_path)
            # Áp dụng normalization
            normalized = transforms.Normalize(
                mean=self.config['preprocessing']['image']['normalize']['mean'],
                std=self.config['preprocessing']['image']['normalize']['std']
            )(tensor)
            return normalized
        except Exception as e:
            print(f"Error loading {processed_path}: {e}")
            return None
