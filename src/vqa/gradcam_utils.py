# src/vqa/gradcam_utils.py
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import ImageDraw

def generate_gradcam(model, target_layers, input_tensor, rgb_img):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam

def draw_bounding_box(image, grayscale_cam, threshold=0.5):
    heatmap = grayscale_cam
    mask = heatmap >= threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
    return image
