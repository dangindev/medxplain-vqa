import os
import argparse
from PIL import Image
from datetime import datetime
import torch
import numpy as np

from vqa.blip2_model import BLIP2VQA
from vqa.gradcam_utils import generate_gradcam, draw_bounding_box
from gemini.rewrite import rewrite_question
from gemini.rationale import generate_rationale  # phần này sẽ tạo ở bước sau

from torchvision import transforms
from pathlib import Path
import json

from vqa.gradcam_utils import BLIP2VisualWrapper

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tensor = transform(image).unsqueeze(0).to(device).float()

    # ⚠️ Sửa tại đây:
    if hasattr(torch.cuda, "is_bf16_supported") and device == "cuda":
        tensor = tensor.half()  # chuyển sang float16 nếu model dùng half
    rgb_image = np.array(image.resize((224, 224))) / 255.0
    return image, tensor, rgb_image


def run_pipeline(image_path, question, save_id="test_output"):
    # Load image
    image, image_tensor, rgb_image = load_and_preprocess_image(image_path)

    # Rewrite question using Gemini
    rewritten_question = rewrite_question(question)

    # Load BLIP2 and answer
    vqa_model = BLIP2VQA(model_name="blip2_opt", model_type="pretrain_opt2.7b")
    answer = vqa_model.answer(image_path, rewritten_question)

    # Grad-CAM
    model = BLIP2VisualWrapper(vqa_model.model)
    model = model.float()
    
    target_layer = model.encoder.blocks[-1].norm1
    cam_image, grayscale_cam = generate_gradcam(model, [target_layer], image_tensor, rgb_image)

    # Bounding Box
    final_image, bbox_coords = draw_bounding_box(Image.fromarray(cam_image), grayscale_cam, return_coords=True)

    # Generate rationale using Gemini
    rationale = generate_rationale(rewritten_question, answer, bbox_coords)

    # Save image
    Path("outputs/combined").mkdir(parents=True, exist_ok=True)
    img_save_path = f"outputs/combined/{save_id}.jpg"
    final_image.save(img_save_path)

    # Save results as JSON
    result = {
        "original_question": question,
        "rewritten_question": rewritten_question,
        "answer": answer,
        "bbox_coords": bbox_coords,
        "rationale": rationale,
        "image_output": img_save_path
    }

    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    with open(f"outputs/results/{save_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved result to outputs/results/{save_id}.json")
    print(f"✅ Saved overlay image to outputs/combined/{save_id}.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--save_id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    run_pipeline(args.image, args.question, args.save_id)
