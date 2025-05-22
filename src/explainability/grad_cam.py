import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Grad-CAM implementation for BLIP model
    Based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    
    def __init__(self, model, layer_name="vision_model.encoder.layers.11"):
        """
        Initialize Grad-CAM with a model and target layer
        
        Args:
            model: BLIP model
            layer_name: Target layer for Grad-CAM (typically the last convolutional layer)
        """
        self.model = model
        self.layer_name = layer_name
        self.device = next(model.parameters()).device
        
        # Đăng ký hooks
        self.gradients = None
        self.activations = None
        self.hooks_registered = False
        
        # Đăng ký hooks
        self._register_hooks()
        
        logger.info(f"Grad-CAM initialized with layer: {layer_name}")
    
    def _register_hooks(self):
        """Đăng ký hooks để lấy gradients và activations"""
        if self.hooks_registered:
            logger.info("Hooks already registered")
            return
        
        # Tìm layer mục tiêu
        target_layer = self._find_target_layer()
        if target_layer is None:
            logger.error(f"Layer {self.layer_name} not found in model")
            return
        
        # Đăng ký forward hook
        def forward_hook(module, input, output):
            self.activations = output
        
        # Đăng ký backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Gắn hooks
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        self.hooks_registered = True
        logger.info("Hooks registered successfully")
    
    def _find_target_layer(self):
        """Tìm layer mục tiêu trong mô hình"""
        # Parse layer name
        if "." not in self.layer_name:
            return getattr(self.model, self.layer_name, None)
        
        # Xử lý nested layers
        parts = self.layer_name.split(".")
        current = self.model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                logger.error(f"Cannot find {part} in {current}")
                return None
        
        return current
    
    def remove_hooks(self):
        """Gỡ bỏ hooks để tránh memory leak"""
        if self.hooks_registered:
            self.forward_handle.remove()
            self.backward_handle.remove()
            self.hooks_registered = False
            logger.info("Hooks removed")
    
    def _preprocess_image(self, image):
        """
        Tiền xử lý hình ảnh nếu cần
        
        Args:
            image: PIL Image hoặc tensor
            
        Returns:
            tensor: Tensor đã xử lý
        """
        if isinstance(image, Image.Image):
            # Nếu dùng processor của BLIP để xử lý, trả về ngay
            return None
        
        if isinstance(image, torch.Tensor):
            # Đã là tensor, đưa lên đúng device
            return image.to(self.device)
        
        # Nếu không phải cả PIL Image và torch.Tensor, báo lỗi
        logger.error(f"Unsupported image type: {type(image)}")
        return None
    
    def _generate_cam(self, width, height):
        """
        Tạo bản đồ Grad-CAM từ gradients và activations
        
        Args:
            width: Chiều rộng của hình ảnh gốc
            height: Chiều cao của hình ảnh gốc
            
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        # Đảm bảo có gradients và activations
        if self.gradients is None or self.activations is None:
            logger.error("Gradients or activations not available")
            return None
        
        # Tính trọng số
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Tạo class activation map
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Chỉ giữ lại giá trị dương
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Chuyển về numpy
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Resize về kích thước hình ảnh gốc
        cam = cv2.resize(cam, (width, height))
        
        # Normalize lại để hiển thị
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam
    
    def __call__(self, image, question, inputs=None, original_size=None):
        """
        Tạo Grad-CAM heatmap cho hình ảnh và câu hỏi
        
        Args:
            image: PIL Image hoặc tensor
            question: Câu hỏi
            inputs: Đầu vào đã xử lý (nếu có)
            original_size: Kích thước gốc của hình ảnh (width, height)
            
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        self.model.eval()
        
        # Xác định kích thước
        if original_size is None:
            if isinstance(image, Image.Image):
                original_size = image.size  # (width, height)
            elif isinstance(image, torch.Tensor) and image.dim() == 3:
                # Tensor shape: C x H x W
                original_size = (image.shape[2], image.shape[1])  # (width, height)
            elif isinstance(image, torch.Tensor) and image.dim() == 4:
                # Tensor shape: B x C x H x W
                original_size = (image.shape[3], image.shape[2])  # (width, height)
        
        if original_size is None:
            logger.error("Cannot determine image size")
            return None
        
        width, height = original_size
        
        # Reset gradients
        self.model.zero_grad()
        
        # Xử lý đầu vào nếu chưa có
        if inputs is None:
            # Xử lý hình ảnh và câu hỏi bằng processor của BLIP
            inputs = self.model.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)
        
        # Forward pass
        try:
            with torch.set_grad_enabled(True):
                outputs = self.model.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values,
                    return_dict=True
                )
                
                # Lấy logits đầu ra
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Tính target score (có thể điều chỉnh tùy theo kiến trúc model)
                if hasattr(logits, 'mean'):
                    target_score = logits.mean()
                else:
                    # Nếu không có logits, dùng cách khác để tính target score
                    target_score = outputs.image_embeddings.mean() if hasattr(outputs, 'image_embeddings') else outputs
                
                # Backward pass
                target_score.backward()
        except Exception as e:
            logger.error(f"Error during forward/backward pass: {e}")
            return None
        
        # Tạo Grad-CAM
        grad_cam = self._generate_cam(width, height)
        
        # Reset self.gradients và self.activations
        self.gradients = None
        self.activations = None
        
        return grad_cam
