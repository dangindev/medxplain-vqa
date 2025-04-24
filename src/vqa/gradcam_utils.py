import torch
import numpy as np
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import ImageDraw

# ✅ Wrapper giúp GradCAM không đụng BLIP2 cấu trúc đặc biệt
class BLIP2VisualWrapper(nn.Module):
    def __init__(self, blip2_model):
        super().__init__()
        self.encoder = blip2_model.visual_encoder.float()  # ép float ở đây luôn

    def forward(self, x):
        return self.encoder(x.float())  # ép input ảnh về float32

# ✅ Tạo heatmap Grad-CAM và overlay lên ảnh
def generate_gradcam(model, target_layers, input_tensor, rgb_img):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam

# ✅ Vẽ bounding box lên ảnh và trả về toạ độ
def draw_bounding_box(image, grayscale_cam, threshold=0.5, return_coords=False):
    heatmap = grayscale_cam
    mask = heatmap >= threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image, None if return_coords else image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline='red', width=2)

    if return_coords:
        return image, {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
    return image
