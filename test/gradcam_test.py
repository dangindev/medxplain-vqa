# test/gradcam_test.py
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from src.vqa.gradcam_utils import generate_gradcam
from src.vqa.gradcam_utils import draw_bounding_box

def main():
    model = models.resnet18(pretrained=True)
    model.eval()

    target_layers = [model.layer4[-1]]

    img_path = 'data/images/test/test_0001.jpg'
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)
    rgb_img = np.array(img.resize((224, 224))) / 255.0

    cam_image, grayscale_cam = generate_gradcam(model, target_layers, input_tensor, rgb_img)

    os.makedirs('outputs/gradcam', exist_ok=True)
    output_path = 'outputs/gradcam/test_0001.jpg'
    Image.fromarray(cam_image).save(output_path)
    print(f"✅ Grad-CAM image saved to {output_path}")

    img_with_box = draw_bounding_box(Image.fromarray(cam_image), grayscale_cam)
    output_box_path = 'outputs/bounding_boxes/test_0001.jpg'
    os.makedirs('outputs/bounding_boxes', exist_ok=True)
    img_with_box.save(output_box_path)
    print(f"✅ Bounding box image saved to {output_box_path}")

if __name__ == '__main__':
    main()