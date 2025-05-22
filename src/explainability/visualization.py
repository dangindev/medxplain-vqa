import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)

def apply_colormap(cam, cmap_name='jet'):
    """Áp dụng colormap cho Grad-CAM heatmap"""
    # Chuyển về dạng uint8
    cam_uint8 = np.uint8(255 * cam)
    
    # Áp dụng colormap
    colored_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    
    # Chuyển về định dạng RGB (từ BGR của OpenCV)
    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
    
    return colored_cam

def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Chồng heatmap lên hình ảnh gốc
    
    Args:
        image: PIL Image hoặc numpy array
        heatmap: Grad-CAM heatmap (numpy array)
        alpha: Độ trong suốt của heatmap (0-1)
        
    Returns:
        numpy.ndarray: Hình ảnh với heatmap được chồng lên
    """
    # Chuyển image về numpy array nếu là PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 4:  # Batch dimension
            image = image[0]  # Lấy hình ảnh đầu tiên trong batch
        
        # Chuyển từ tensor về numpy và chuyển kênh về dạng HWC
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Đảm bảo giá trị nằm trong khoảng [0, 1]
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
    else:
        image_np = image
    
    # Nếu image là grayscale, chuyển sang RGB
    if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    # Đảm bảo giá trị nằm trong khoảng [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Tạo colormap cho heatmap
    colored_heatmap = apply_colormap(heatmap)
    
    # Chồng heatmap lên hình ảnh
    overlay = image_np * (1 - alpha) + colored_heatmap / 255.0 * alpha
    
    # Clip giá trị về khoảng [0, 1]
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def visualize_gradcam(image, cam, alpha=0.5, return_fig=False):
    """
    Hiển thị Grad-CAM
    
    Args:
        image: PIL Image hoặc numpy array
        cam: Grad-CAM heatmap
        alpha: Độ trong suốt của heatmap
        return_fig: Có trả về đối tượng figure hay không
        
    Returns:
        matplotlib.figure.Figure (nếu return_fig=True)
    """
    plt.figure(figsize=(12, 5))
    
    # Hiển thị hình ảnh gốc
    plt.subplot(1, 3, 1)
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # Batch dimension
            image = image[0]  # Lấy hình ảnh đầu tiên trong batch
        
        # Chuyển từ tensor về numpy và chuyển kênh về dạng HWC
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Đảm bảo giá trị nằm trong khoảng [0, 1]
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        plt.imshow(image_np)
    else:
        plt.imshow(np.array(image))
    
    plt.title("Original Image")
    plt.axis("off")
    
    # Hiển thị Grad-CAM
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM")
    plt.axis("off")
    
    # Hiển thị overlay
    plt.subplot(1, 3, 3)
    overlay = overlay_heatmap(image, cam, alpha)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    if return_fig:
        return plt.gcf()
    
    plt.show()

def save_gradcam_visualization(image, cam, output_path, alpha=0.5):
    """
    Lưu trực quan hóa Grad-CAM
    
    Args:
        image: PIL Image hoặc numpy array
        cam: Grad-CAM heatmap
        output_path: Đường dẫn để lưu
        alpha: Độ trong suốt của heatmap
    """
    try:
        fig = visualize_gradcam(image, cam, alpha, return_fig=True)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Grad-CAM visualization saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving Grad-CAM visualization: {e}")
        return False

def get_salient_regions(cam, threshold=0.5):
    """
    Trích xuất các vùng nổi bật từ Grad-CAM
    
    Args:
        cam: Grad-CAM heatmap
        threshold: Ngưỡng để xác định vùng nổi bật
        
    Returns:
        list: Danh sách các vùng nổi bật dưới dạng (x, y, w, h, score)
    """
    # Chuyển về dạng uint8 và áp dụng ngưỡng
    cam_uint8 = np.uint8(255 * cam)
    thresh = np.uint8(cam > threshold)
    
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trích xuất bounding box và score trung bình
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Tính điểm trung bình trong vùng
        region_mask = np.zeros_like(cam)
        cv2.drawContours(region_mask, [contour], 0, 1, -1)
        mean_score = np.mean(cam * region_mask)
        
        regions.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'score': float(mean_score)
        })
    
    # Sắp xếp theo score giảm dần
    regions.sort(key=lambda x: x['score'], reverse=True)
    
    return regions

def describe_salient_regions(regions, image_width, image_height):
    """
    Mô tả các vùng nổi bật bằng văn bản
    
    Args:
        regions: Danh sách các vùng nổi bật
        image_width: Chiều rộng hình ảnh
        image_height: Chiều cao hình ảnh
        
    Returns:
        str: Mô tả các vùng nổi bật
    """
    if not regions:
        return "No significant regions were identified."
    
    descriptions = []
    
    for i, region in enumerate(regions[:3]):  # Chỉ mô tả top 3 vùng
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        score = region['score']
        
        # Tính vị trí tương đối
        center_x = x + w/2
        center_y = y + h/2
        
        # Xác định vị trí
        horizontal_pos = "left" if center_x < image_width/3 else "right" if center_x > 2*image_width/3 else "center"
        vertical_pos = "top" if center_y < image_height/3 else "bottom" if center_y > 2*image_height/3 else "middle"
        
        # Đánh giá kích thước
        area_ratio = (w * h) / (image_width * image_height)
        size_desc = "small" if area_ratio < 0.1 else "large" if area_ratio > 0.3 else "medium"
        
        # Tạo mô tả
        region_desc = f"Region {i+1} (importance: {score:.2f}): A {size_desc} area in the {vertical_pos} {horizontal_pos} portion of the image."
        descriptions.append(region_desc)
    
    return "\n".join(descriptions)
