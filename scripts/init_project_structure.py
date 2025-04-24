import os

folders = [
    "env",
    "data/raw",
    "data/images/train",
    "data/images/val",
    "data/images/test",
    "data/questions",
    "src/utils",
    "src/vqa",
    "src/cam",
    "src/gemini",
    "src/evaluate",
    "test",
    "scripts"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}")

# Tạo file README mặc định
readme_path = "README.md"
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write("# MedXplain-VQA\n\nA reproducible explainable VQA framework for medical imaging.")
    print(f"✅ Created: {readme_path}")
