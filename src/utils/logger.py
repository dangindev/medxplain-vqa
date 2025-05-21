import logging
import os
from datetime import datetime

def setup_logger(name, log_dir, level=logging.INFO):
    """Thiết lập logger với file handler và stream handler"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Định dạng logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Tạo thư mục log nếu chưa tồn tại
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{now}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger
