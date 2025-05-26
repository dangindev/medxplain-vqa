# scripts/fix_gradcam_and_test.py
#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

class RobustGradCAM:
    """
    Robust Grad-CAM implementation that works consistently
    """
    
    def __init__(self, model, layer_name="vision_model.encoder.layers.11"):
        self.model = model
        self.layer_name = layer_name
        self.device = next(model.parameters()).device
        
        # Hook storage
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks with retry mechanism
        self._register_hooks_robust()
    
    def _register_hooks_robust(self):
        """Register hooks with multiple fallback strategies"""
        
        # Strategy 1: Try exact layer name
        target_layer = self._find_layer_robust(self.layer_name)
        
        if target_layer is None:
            # Strategy 2: Try alternative layer names
            alternative_layers = [
                "vision_model.encoder.layers.10",
                "vision_model.encoder.layers.9", 
                "vision_model.encoder.layer.11",
                "vision_model.encoder.layer.10"
            ]
            
            for alt_layer in alternative_layers:
                target_layer = self._find_layer_robust(alt_layer)
                if target_layer is not None:
                    print(f"✅ Using alternative layer: {alt_layer}")
                    break
        
        if target_layer is None:
            print("❌ Could not find any suitable layer for Grad-CAM")
            return False
        
        print(f"✅ Target layer found: {target_layer}")
        
        # Register hooks
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
            print(f"✅ Forward hook captured: {self.activations.shape}")
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
            elif grad_output is not None:
                self.gradients = grad_output.detach()
            
            if self.gradients is not None:
                print(f"✅ Backward hook captured: {self.gradients.shape}")
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        self.hooks = [forward_handle, backward_handle]
        return True
    
    def _find_layer_robust(self, layer_name):
        """Robust layer finding with detailed logging"""
        try:
            parts = layer_name.split(".")
            current = self.model
            
            for i, part in enumerate(parts):
                if hasattr(current, part):
                    current = getattr(current, part)
                    print(f"✅ Found part {i}: {part}")
                else:
                    print(f"❌ Cannot find part {i}: {part}")
                    if hasattr(current, '_modules'):
                        available = list(current._modules.keys())
                        print(f"Available modules: {available}")
                    return None
            
            return current
            
        except Exception as e:
            print(f"❌ Error finding layer {layer_name}: {e}")
            return None
    
    def generate_cam(self, image, question):
        """Generate Grad-CAM with robust error handling"""
        
        # Reset
        self.gradients = None
        self.activations = None
        self.model.zero_grad()
        
        # Process inputs
        if hasattr(self.model, 'processor'):
            processor = self.model.processor
        else:
            print("❌ Model has no processor")
            return None
        
        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device)
        
        print(f"Input processed: {[(k, v.shape) for k, v in inputs.items() if hasattr(v, 'shape')]}")
        
        # Forward pass with multiple strategies
        try:
            with torch.set_grad_enabled(True):
                # Strategy 1: Vision model only
                vision_outputs = self.model.vision_model(inputs.pixel_values)
                
                # Get target for backward
                if hasattr(vision_outputs, 'last_hidden_state'):
                    target = vision_outputs.last_hidden_state.mean()
                elif hasattr(vision_outputs, 'pooler_output'):
                    target = vision_outputs.pooler_output.mean()
                else:
                    print("❌ Cannot find suitable target")
                    return None
                
                print(f"Target for backward: {target}")
                
                # Backward pass
                target.backward()
                
                # Check if hooks captured data
                if self.gradients is None or self.activations is None:
                    print(f"❌ Hooks failed - Gradients: {self.gradients is not None}, Activations: {self.activations is not None}")
                    return None
                
                # Generate CAM
                return self._compute_cam(image.size)
                
        except Exception as e:
            print(f"❌ Forward/backward error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_cam(self, image_size):
        """Compute CAM from gradients and activations"""
        
        print(f"Computing CAM from gradients: {self.gradients.shape}, activations: {self.activations.shape}")
        
        # Handle different shapes
        if len(self.gradients.shape) == 3:  # [batch, seq_len, hidden_dim]
            weights = torch.mean(self.gradients, dim=(0, 1))  # [hidden_dim]
            activations = self.activations[0]  # [seq_len, hidden_dim]
            
            cam = torch.sum(activations * weights.unsqueeze(0), dim=1)  # [seq_len]
            
            # Reshape to spatial
            seq_len = cam.shape[0]
            
            # Try different spatial sizes
            for spatial_size in [14, 16, 12]:  # Common patch grid sizes
                if spatial_size * spatial_size == seq_len:
                    cam_spatial = cam.reshape(spatial_size, spatial_size)
                    break
                elif spatial_size * spatial_size == seq_len - 1:  # With CLS token
                    cam_spatial = cam[1:].reshape(spatial_size, spatial_size)
                    break
            else:
                # Fallback
                spatial_size = int(np.sqrt(seq_len))
                cam_spatial = cam[:spatial_size*spatial_size].reshape(spatial_size, spatial_size)
            
        elif len(self.gradients.shape) == 4:  # [batch, height, width, hidden_dim]
            weights = torch.mean(self.gradients, dim=(0, 1, 2))
            activations = self.activations[0]
            cam_spatial = torch.sum(activations * weights, dim=2)
        
        else:
            print(f"❌ Unexpected gradient shape: {self.gradients.shape}")
            return None
        
        # Apply ReLU and normalize
        cam_spatial = torch.relu(cam_spatial)
        if torch.max(cam_spatial) > 0:
            cam_spatial = cam_spatial / torch.max(cam_spatial)
        
        # Convert to numpy and resize
        cam = cam_spatial.cpu().detach().numpy()
        
        import cv2
        cam = cv2.resize(cam, image_size)
        
        # Final normalization
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        print(f"✅ CAM generated: {cam.shape}, range: [{cam.min():.3f}, {cam.max():.3f}]")
        return cam
    
    def remove_hooks(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def test_robust_gradcam(config_path, model_path, image_path, question):
    """Test the robust Grad-CAM implementation"""
    
    # Load config and model
    config = Config(config_path)
    logger = setup_logger('robust_gradcam_test', 'logs', level='INFO')
    
    # Load BLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    blip_model = BLIP2VQA(config, train_mode=False)
    blip_model.device = device
    
    if os.path.isdir(model_path):
        blip_model.model = type(blip_model.model).from_pretrained(model_path)
        blip_model.model.to(device)
    
    blip_model.model.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Test robust Grad-CAM
    print("🔧 Testing Robust Grad-CAM...")
    robust_gradcam = RobustGradCAM(blip_model.model)
    
    # Generate CAM
    cam = robust_gradcam.generate_cam(image, question)
    
    if cam is not None:
        print("✅ Robust Grad-CAM successful!")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, alpha=0.6)
        axes[2].imshow(cam, cmap='jet', alpha=0.4)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.suptitle(f"Question: {question}")
        plt.tight_layout()
        
        output_file = "robust_gradcam_test.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to {output_file}")
        
        # Clean up
        robust_gradcam.remove_hooks()
        
        return True
    else:
        print("❌ Robust Grad-CAM failed")
        robust_gradcam.remove_hooks()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Robust Grad-CAM')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--model-path', default='checkpoints/blip/checkpoints/best_hf_model')
    parser.add_argument('--image', default='data/images/test/test_5238.jpg')
    parser.add_argument('--question', default='what does this image show?')
    
    args = parser.parse_args()
    
    success = test_robust_gradcam(args.config, args.model_path, args.image, args.question)
    
    if success:
        print("\n🎉 Robust Grad-CAM test PASSED!")
        print("Now you can integrate this into medxplain_vqa.py")
    else:
        print("\n❌ Robust Grad-CAM test FAILED!")
        print("Need further debugging...")

if __name__ == "__main__":
    main()
