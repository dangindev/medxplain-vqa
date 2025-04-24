import os
import json
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

dataset = load_from_disk("data/raw/pathvqa")

# Tạo thư mục output
for split_name in ["train", "val", "test"]:
    os.makedirs(f"data/images/{split_name}", exist_ok=True)
os.makedirs("data/questions", exist_ok=True)

# Ánh xạ tên split trong Hugging Face
split_map = {
    "train": "train",
    "val": "validation",
    "test": "test"
}

def save_image_and_qas(split_name):
    split_data = dataset[split_map[split_name]]
    questions = []

    for idx, sample in tqdm(enumerate(split_data), total=len(split_data), desc=f"Processing {split_name}"):
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        image_id = f"{split_name}_{idx:04d}"
        img_path = f"data/images/{split_name}/{image_id}.jpg"

        # Resize và lưu ảnh
        image.convert("RGB").resize((224, 224)).save(img_path)

        # Lưu QA
        questions.append({
            "image_id": image_id,
            "question": question,
            "answer": answer
        })

    # Lưu câu hỏi
    with open(f"data/questions/{split_name}.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"✅ Processed {split_name}: {len(questions)} samples")

# Chạy cho cả 3 split
save_image_and_qas("train")
save_image_and_qas("val")
save_image_and_qas("test")
