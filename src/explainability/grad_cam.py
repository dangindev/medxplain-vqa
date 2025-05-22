import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Grad-CAM implementation for BLIP model with proper tuple handling
    Based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    
    def __init__(self, model, layer_name="vision_model.encoder.layers.11"):
        """
        Initialize Grad-CAM with a model and target layer
        
        Args:
            model: BLIP model (BlipForQuestionAnswering or BLIP2VQA wrapper)
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
        
        logger.info(f"Found target layer: {target_layer}")
        
        # Đăng ký forward hook
        def forward_hook(module, input, output):
            # Handle tuple output from BLIP layers
            if isinstance(output, tuple):
                # BLIP encoder layers return (hidden_states, attention_weights, ...)
                # We want the hidden states (first element)
                self.activations = output[0]
                logger.debug(f"Forward hook captured activations from tuple: {output[0].shape}")
            else:
                self.activations = output
                logger.debug(f"Forward hook captured activations from tensor: {output.shape}")
        
        # Đăng ký backward hook
        def backward_hook(module, grad_input, grad_output):
            # Handle tuple gradients
            if isinstance(grad_output, tuple):
                # Take the first gradient (corresponding to hidden states)
                if grad_output[0] is not None:
                    self.gradients = grad_output[0]
                    logger.debug(f"Backward hook captured gradients from tuple: {grad_output[0].shape}")
            else:
                if grad_output is not None:
                    self.gradients = grad_output
                    logger.debug(f"Backward hook captured gradients from tensor: {grad_output.shape}")
        
        # Gắn hooks
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        self.hooks_registered = True
        logger.info("Hooks registered successfully")
    
    def _find_target_layer(self):
        """Tìm layer mục tiêu trong mô hình"""
        logger.info(f"Looking for layer: {self.layer_name}")
        
        # Parse layer name
        if "." not in self.layer_name:
            layer = getattr(self.model, self.layer_name, None)
            logger.info(f"Found simple layer: {layer}")
            return layer
        
        # Xử lý nested layers
        parts = self.layer_name.split(".")
        current = self.model
        
        for i, part in enumerate(parts):
            if hasattr(current, part):
                current = getattr(current, part)
                logger.debug(f"Step {i}: Found {part} -> {type(current)}")
            else:
                logger.error(f"Cannot find {part} in {current}")
                logger.error(f"Available attributes: {list(current._modules.keys()) if hasattr(current, '_modules') else 'No _modules'}")
                return None
        
        logger.info(f"Final target layer found: {type(current)}")
        return current
    
    def remove_hooks(self):
        """Gỡ bỏ hooks để tránh memory leak"""
        if self.hooks_registered:
            self.forward_handle.remove()
            self.backward_handle.remove()
            self.hooks_registered = False
            logger.info("Hooks removed")
    
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
            logger.error(f"Gradients: {self.gradients}")
            logger.error(f"Activations: {self.activations}")
            return None
        
        logger.info(f"Generating CAM from gradients: {self.gradients.shape}, activations: {self.activations.shape}")
        
        # Handle different tensor shapes from BLIP
        if len(self.gradients.shape) == 3:  # [batch, seq_len, hidden_dim]
            # Average over batch and compute weights
            weights = torch.mean(self.gradients, dim=(0, 1))  # [hidden_dim]
            activations = self.activations[0]  # Take first batch item [seq_len, hidden_dim]
            
            # Compute weighted sum
            cam = torch.sum(activations * weights.unsqueeze(0), dim=1)  # [seq_len]
            
            # Reshape to spatial dimensions
            # For BLIP vision, sequence length should be (H/patch_size) * (W/patch_size)
            seq_len = cam.shape[0]
            
            # Try to infer spatial dimensions (14x14 for 224x224 input with 16x16 patches)
            spatial_size = int(np.sqrt(seq_len - 1))  # -1 for potential CLS token
            if spatial_size * spatial_size == seq_len - 1:
                # Remove CLS token and reshape
                cam_spatial = cam[1:].reshape(spatial_size, spatial_size)
            elif spatial_size * spatial_size == seq_len:
                cam_spatial = cam.reshape(spatial_size, spatial_size)
            else:
                # Fallback: assume square
                spatial_size = int(np.sqrt(seq_len))
                cam_spatial = cam[:spatial_size*spatial_size].reshape(spatial_size, spatial_size)
            
            logger.debug(f"Reshaped CAM to spatial: {cam_spatial.shape}")
            
        elif len(self.gradients.shape) == 4:  # [batch, height, width, hidden_dim]
            weights = torch.mean(self.gradients, dim=(0, 1, 2))  # [hidden_dim]
            activations = self.activations[0]  # [height, width, hidden_dim]
            cam_spatial = torch.sum(activations * weights, dim=2)  # [height, width]
        
        else:
            logger.error(f"Unexpected gradient shape: {self.gradients.shape}")
            return None
        
        # Apply ReLU
        cam_spatial = F.relu(cam_spatial)
        
        # Normalize
        if torch.max(cam_spatial) > 0:
            cam_spatial = cam_spatial / torch.max(cam_spatial)
        
        # Chuyển về numpy
        cam = cam_spatial.cpu().detach().numpy()
        
        # Resize về kích thước hình ảnh gốc
        cam = cv2.resize(cam, (width, height))
        
        # Normalize lại để hiển thị
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        logger.info(f"Generated CAM with shape: {cam.shape}, min: {np.min(cam)}, max: {np.max(cam)}")
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
        logger.info("Starting Grad-CAM generation")
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
        logger.info(f"Target size: {width}x{height}")
        
        # Reset gradients
        self.model.zero_grad()
        
        # Xử lý đầu vào nếu chưa có
        if inputs is None:
            # Check if model has processor attribute
            if hasattr(self.model, 'processor'):
                processor = self.model.processor
            else:
                logger.error("Model does not have processor attribute")
                return None
                
            # Xử lý hình ảnh và câu hỏi bằng processor
            inputs = processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)
        
        logger.info(f"Input shapes: {[(k, v.shape) for k, v in inputs.items() if hasattr(v, 'shape')]}")
        
        # Forward pass using vision model approach
        try:
            with torch.set_grad_enabled(True):
                # Call vision model to trigger hooks
                vision_outputs = self.model.vision_model(inputs.pixel_values)
                
                # Get suitable target for backward pass
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    vision_features = vision_outputs.pooler_output
                    logger.info(f"Using pooler_output: {vision_features.shape}")
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_features = vision_outputs.last_hidden_state
                    logger.info(f"Using last_hidden_state: {vision_features.shape}")
                    # Take mean over sequence dimension for vision
                    vision_features = vision_features.mean(dim=1)  # [batch, hidden_dim]
                else:
                    logger.error("Cannot find suitable vision features")
                    return None
                
                # Create target for backward pass
                target_score = vision_features.mean()
                logger.info(f"Target score: {target_score}")
                
                # Backward pass
                logger.info("Starting backward pass")
                target_score.backward()
                logger.info("Backward pass completed")
                
        except Exception as e:
            logger.error(f"Error during forward/backward pass: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        
        # Tạo Grad-CAM
        logger.info("Generating CAM from gradients and activations")
        grad_cam = self._generate_cam(width, height)
        
        # Reset self.gradients và self.activations
        self.gradients = None
        self.activations = None
        
        if grad_cam is not None:
            logger.info("Grad-CAM generation successful")
        else:
            logger.error("Grad-CAM generation failed")
        
        return grad_cam
