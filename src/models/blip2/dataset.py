import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PathVQAFineTuneDataset(Dataset):
    """
    Dataset cho việc fine-tune BLIP trên dữ liệu PathVQA
    """
    def __init__(self, config, processor, split='train'):
        self.config = config
        self.processor = processor
        self.split = split
        
        # Đường dẫn dữ liệu
        self.questions_file = config['data'][f'{split}_questions']
        self.images_dir = config['data'][f'{split}_images']
        
        # Đọc dữ liệu
        logger.info(f"Loading {split} split data...")
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} samples from {split} split")
        
        # Tạo mapping từ image_id đến đường dẫn ảnh
        self.image_paths = self._get_image_paths()
    
    def _load_data(self):
        """Tải dữ liệu câu hỏi và câu trả lời"""
        data = []
        
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        
        return data
    
    def _get_image_paths(self):
        """Tạo mapping từ image_id đến đường dẫn ảnh"""
        image_paths = {}
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        for item in self.data:
            image_id = item['image_id']
            
            # Tìm file ảnh phù hợp
            for ext in image_extensions:
                img_path = os.path.join(self.images_dir, f"{image_id}{ext}")
                if os.path.exists(img_path):
                    image_paths[image_id] = img_path
                    break
            
            # Kiểm tra nếu không tìm thấy ảnh
            if image_id not in image_paths:
                # Tìm kiếm dựa trên tên file
                for ext in image_extensions:
                    potential_imgs = list(Path(self.images_dir).glob(f"{image_id}*{ext}"))
                    if potential_imgs:
                        image_paths[image_id] = str(potential_imgs[0])
                        break
        
        # Log số lượng ảnh tìm thấy
        found = len(image_paths)
        total = len(self.data)
        logger.info(f"Found {found}/{total} images for {self.split} split")
        
        return image_paths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu"""
        item = self.data[idx]
        image_id = item['image_id']
        question = item['question']
        answer = item['answer']
        
        # Tìm và mở ảnh
        if image_id in self.image_paths:
            try:
                image = Image.open(self.image_paths[image_id]).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {image_id}: {e}")
                # Trả về một mẫu khác nếu gặp lỗi
                return self.__getitem__((idx + 1) % len(self))
        else:
            logger.warning(f"Image {image_id} not found. Using a different sample.")
            return self.__getitem__((idx + 1) % len(self))
        
        # Xử lý đầu vào bằng processor
        inputs = self.processor(image, text=question, return_tensors="pt")
        
        # Xử lý câu trả lời
        # Đảm bảo câu trả lời không quá dài
        max_answer_length = self.config['model']['blip2']['max_answer_length']
        answer = answer[:max_answer_length] if len(answer) > max_answer_length else answer
        
        # Tokenize câu trả lời
        answer_tokens = self.processor.tokenizer(
            answer,
            max_length=max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        # Tạo labels
        labels = answer_tokens.input_ids.squeeze(0)
        
        # Chuẩn bị dữ liệu trả về
        encoding = {
            'pixel_values': inputs.pixel_values,
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': labels,
            'image_id': image_id,
            'question_text': question,
            'answer_text': answer
        }
        
        return encoding

def pad_sequence(sequences, batch_first=True, padding_value=0):
    """Pad sequence có độ dài khác nhau để có thể batch"""
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
        
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
            
    return out_tensor

def get_dataloader(config, processor, split='train', batch_size=None, shuffle=None, num_workers=None):
    """Tạo dataloader cho fine-tuning"""
    if batch_size is None:
        batch_size = config['training']['batch_size'] if split == 'train' else config['training']['val_batch_size']
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    if num_workers is None:
        num_workers = config['training']['num_workers']
    
    dataset = PathVQAFineTuneDataset(config, processor, split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset

def collate_fn(batch):
    """Hàm collate để xử lý batch"""
    # Loại bỏ các mẫu None (nếu có)
    batch = [b for b in batch if b is not None]
    
    if not batch:
        return None
    
    # Khởi tạo dict cho batch
    batch_dict = {
        'image_id': [],
        'question_text': [],
        'answer_text': []
    }
    
    # Xử lý pixel_values
    if 'pixel_values' in batch[0]:
        batch_dict['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])
    
    # Xử lý input_ids và attention_mask với padding
    if 'input_ids' in batch[0]:
        input_ids = [item['input_ids'] for item in batch]
        batch_dict['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
    
    if 'attention_mask' in batch[0]:
        attention_masks = [item['attention_mask'] for item in batch]
        batch_dict['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
    
    # Xử lý labels với padding
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        batch_dict['labels'] = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # -100 là giá trị bỏ qua trong loss
        )
    
    # Thêm các thông tin không phải tensor
    for item in batch:
        batch_dict['image_id'].append(item['image_id'])
        batch_dict['question_text'].append(item['question_text'])
        batch_dict['answer_text'].append(item['answer_text'])
    
    return batch_dict
