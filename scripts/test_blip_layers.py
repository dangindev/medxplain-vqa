#!/usr/bin/env python
import os
import sys
import torch
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.models.blip2.model import BLIP2VQA

def explore_model_structure(model, prefix="", max_depth=3, current_depth=0):
    """Khám phá cấu trúc mô hình để tìm layers phù hợp cho Grad-CAM"""
    if current_depth >= max_depth:
        return
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"{'  ' * current_depth}{full_name}: {type(module).__name__}")
        
        # Kiểm tra xem có phải vision-related layer không
        if any(keyword in name.lower() for keyword in ['vision', 'encoder', 'layer']):
            print(f"{'  ' * current_depth}  -> POTENTIAL TARGET: {full_name}")
        
        # Recursive explore
        if hasattr(module, 'named_children') and current_depth < max_depth - 1:
            explore_model_structure(module, full_name, max_depth, current_depth + 1)

def main():
    # Load config
    config = Config('configs/config.yaml')
    
    print("Loading BLIP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    blip_model = BLIP2VQA(config, train_mode=False)
    blip_model.device = device
    
    # Load từ checkpoint
    model_path = 'checkpoints/blip/checkpoints/best_hf_model'
    if os.path.isdir(model_path):
        blip_model.model = type(blip_model.model).from_pretrained(model_path)
        blip_model.model.to(device)
    
    print("=== BLIP Model Structure ===")
    explore_model_structure(blip_model.model, max_depth=4)
    
    print("\n=== Vision Model Structure (if exists) ===")
    if hasattr(blip_model.model, 'vision_model'):
        explore_model_structure(blip_model.model.vision_model, prefix="vision_model", max_depth=3)
    else:
        print("No vision_model found")
    
    print("\n=== Alternative layer suggestions ===")
    potential_layers = [
        "vision_model.encoder.layers.10",
        "vision_model.encoder.layers.9", 
        "vision_model.encoder.layers.8",
        "vision_model.pooler",
        "vision_model.post_layernorm"
    ]
    
    for layer_name in potential_layers:
        try:
            parts = layer_name.split(".")
            current = blip_model.model
            for part in parts:
                current = getattr(current, part)
            print(f"✅ {layer_name}: {type(current).__name__}")
        except AttributeError as e:
            print(f"❌ {layer_name}: Not found")

if __name__ == "__main__":
    main()
