import os
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Chuyển đổi các đường dẫn thành đường dẫn tuyệt đối
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        for section in ['data', 'logging', 'model']:
            if section in self.config:
                for key, value in self.config[section].items():
                    if isinstance(value, str) and ('dir' in key or 'path' in key):
                        if not os.path.isabs(value):
                            self.config[section][key] = str(project_root / value)
        
        # Đảm bảo các thư mục tồn tại
        os.makedirs(self.config['logging']['save_dir'], exist_ok=True)
        os.makedirs(self.config['model']['blip2']['cache_dir'], exist_ok=True)
        if 'processed_dir' in self.config['data']:
            os.makedirs(self.config['data']['processed_dir'], exist_ok=True)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k not in value:
                return default
            value = value[k]
        return value

def load_api_keys(api_key_path):
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            return yaml.safe_load(f)
    return {}
