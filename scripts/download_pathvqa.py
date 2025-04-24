from datasets import load_dataset
import os

# Tải và lưu vào thư mục raw
dataset = load_dataset("flaviagiammarino/path-vqa")
dataset.save_to_disk("data/raw/pathvqa")

print("✅ Dataset downloaded and saved to data/raw/pathvqa")
