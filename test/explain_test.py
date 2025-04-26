import torch, os, json, numpy as np
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from src.vqa.explain import generate_explanation

def main():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()
    target_layers=[model.layer4[-1]]

    img_path="data/images/test/test_0001.jpg"
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)

    mention = generate_explanation({
        "image_path": img_path,
        "answer": "probe",
        "model": model,
        "target_layers": target_layers,
        "input_tensor": input_tensor
    })

    assert os.path.exists(mention["cam_path"])
    assert os.path.exists(mention["cam_npy"])
    assert os.path.exists(mention["bbox_path"])
    print("✅ explain pipeline OK:", json.dumps(mention, indent=2)[:200])

if __name__ == "__main__":
    main()
