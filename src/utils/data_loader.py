import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image

class PathVQADataset(Dataset):
    def __init__(self, config, split='train', transform=None, processed_data=True):
        """
        PathVQA Dataset
        Args:
            config: Cấu hình
            split: 'train', 'val', hoặc 'test'
            transform: Các phép biến đổi cho hình ảnh
            processed_data: Sử dụng dữ liệu đã được tiền xử lý hay không
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.processed_data = processed_data
        
        # Đường dẫn dữ liệu
        if processed_data:
            # Sử dụng dữ liệu đã xử lý
            processed_dir = Path(config['data']['processed_dir'])
            mapping_file = processed_dir / f"{split}_mapping.json"
            
            if not mapping_file.exists():
                raise FileNotFoundError(f"Mapping file not found: {mapping_file}. Run preprocessing first.")
                
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            
            self.questions_file = mapping['processed_questions_file']
            self.images_dir = mapping['processed_images_dir']
        else:
            # Sử dụng dữ liệu gốc
            self.questions_file = config['data'][f'{split}_questions']
            self.images_dir = config['data'][f'{split}_images']
        
        # Tải dữ liệu câu hỏi
        self.qa_data = self._load_qa_data()
        
        # Tạo ánh xạ image_id đến đường dẫn hình ảnh
        self.image_paths = self._get_image_paths()
    
    def _load_qa_data(self):
        """Tải dữ liệu câu hỏi và câu trả lời"""
        qa_data = []
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                qa_data.append(json.loads(line))
        return qa_data
    
    def _get_image_paths(self):
        """Tạo ánh xạ từ image_id đến đường dẫn file"""
        image_paths = {}
        
        if self.processed_data:
            # Dữ liệu đã xử lý - image được lưu dưới dạng .pt files
            for item in self.qa_data:
                image_id = item['image_id']
                image_paths[image_id] = str(Path(self.images_dir) / f"{image_id}.pt")
        else:
            # Dữ liệu gốc - cần quét thư mục hình ảnh
            image_formats = ['.jpg', '.jpeg', '.png']
            for image_format in image_formats:
                for image_path in Path(self.images_dir).glob(f"*{image_format}"):
                    image_id = image_path.stem
                    image_paths[image_id] = str(image_path)
        
        return image_paths
    
    def _load_image(self, image_id):
        """Tải hình ảnh từ đường dẫn"""
        image_path = self.image_paths.get(image_id)
        
        if image_path is None:
            raise ValueError(f"Image with ID {image_id} not found")
        
        if self.processed_data:
            # Dữ liệu đã xử lý - tải tensor trực tiếp
            image = torch.load(image_path)
        else:
            # Dữ liệu gốc - tải và biến đổi hình ảnh
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        return image
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu"""
        qa_item = self.qa_data[idx]
        
        # Lấy hình ảnh
        try:
            image = self._load_image(qa_item['image_id'])
        except Exception as e:
            print(f"Error loading image for {qa_item['image_id']}: {e}")
            # Trả về một mẫu hình ảnh ngẫu nhiên nếu không tải được
            random_idx = np.random.randint(0, len(self))
            qa_item = self.qa_data[random_idx]
            image = self._load_image(qa_item['image_id'])
        
        # Trả về thông tin câu hỏi, câu trả lời và hình ảnh
        return {
            'image_id': qa_item['image_id'],
            'question': qa_item['question'],
            'answer': qa_item['answer'],
            'image': image
        }

def get_data_loader(config, split='train', batch_size=None, shuffle=None, num_workers=None, transform=None):
    """Tạo dataloader cho PathVQA"""
    if batch_size is None:
        batch_size = config['training']['batch_size'] if split == 'train' else config['training']['val_batch_size']
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    if num_workers is None:
        num_workers = config['training']['num_workers']
    
    dataset = PathVQADataset(config, split, transform, processed_data=True)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, dataset
