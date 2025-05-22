#!/usr/bin/env python
import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.blip2.model import BLIP2VQA

class SimpleGradCAM:
    """Simplified Grad-CAM for BLIP with proper tuple handling"""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
    def register_hooks(self):
        """Register hooks on target layer"""
        try:
            # Find target layer
            parts = self.layer_name.split(".")
            current = self.model
            for part in parts:
                current = getattr(current, part)
            
            def forward_hook(module, input, output):
                # Handle tuple output from BLIP layers
                if isinstance(output, tuple):
                    # BLIP encoder layers return (hidden_states, attention_weights, ...)
                    # We want the hidden states (first element)
                    self.activations = output[0].detach()
                    print(f"✅ Captured activations from tuple: {output[0].shape}")
                else:
                    self.activations = output.detach()
                    print(f"✅ Captured activations from tensor: {output.shape}")
                
            def backward_hook(module, grad_input, grad_output):
                # Handle tuple gradients
                if isinstance(grad_output, tuple):
                    # Take the first gradient (corresponding to hidden states)
                    if grad_output[0] is not None:
                        self.gradients = grad_output[0].detach()
                        print(f"✅ Captured gradients from tuple: {grad_output[0].shape}")
                else:
                    self.gradients = grad_output.detach()
                    print(f"✅ Captured gradients from tensor: {grad_output.shape}")
            
            # Register hooks
            h1 = current.register_forward_hook(forward_hook)
            h2 = current.register_full_backward_hook(backward_hook)
            self.hook_handles = [h1, h2]
            
            print(f"✅ Hooks registered on {self.layer_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to register hooks: {e}")
            return False
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def generate_cam_from_vision(self, inputs, image_size):
        """Generate CAM using vision model approach"""
        try:
            self.model.zero_grad()
            
            with torch.set_grad_enabled(True):
                # Call vision model and get output
                vision_outputs = self.model.vision_model(inputs.pixel_values)
                
                # Get the pooled output or last hidden state
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    vision_features = vision_outputs.pooler_output
                    print(f"Using pooler_output: {vision_features.shape}")
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_features = vision_outputs.last_hidden_state
                    print(f"Using last_hidden_state: {vision_features.shape}")
                    # Take mean over sequence dimension for vision
                    vision_features = vision_features.mean(dim=1)  # [batch, hidden_dim]
                else:
                    print("❌ Cannot find suitable vision features")
                    return None
                
                # Create target for backward pass
                target = vision_features.mean()
                print(f"Target for backward: {target}")
                
                # Backward pass
                target.backward()
                
                if self.gradients is not None and self.activations is not None:
                    print(f"Generating CAM from gradients: {self.gradients.shape}, activations: {self.activations.shape}")
                    
                    # Generate CAM - handle different dimensionalities
                    if len(self.gradients.shape) == 3:  # [batch, seq_len, hidden_dim]
                        # Average over batch and compute weights
                        weights = torch.mean(self.gradients, dim=(0, 1))  # [hidden_dim]
                        activations = self.activations[0]  # Take first batch item [seq_len, hidden_dim]
                        
                        # Compute weighted sum
                        cam = torch.sum(activations * weights.unsqueeze(0), dim=1)  # [seq_len]
                        
                        # Reshape to spatial dimensions if needed
                        # For BLIP vision, sequence length should be (H/patch_size) * (W/patch_size)
                        # Assuming 224x224 input with 16x16 patches = 14x14 = 196 tokens
                        seq_len = cam.shape[0]
                        
                        # Try to infer spatial dimensions
                        spatial_size = int(np.sqrt(seq_len - 1))  # -1 for CLS token potentially
                        if spatial_size * spatial_size == seq_len - 1:
                            # Remove CLS token and reshape
                            cam_spatial = cam[1:].reshape(spatial_size, spatial_size)
                        elif spatial_size * spatial_size == seq_len:
                            cam_spatial = cam.reshape(spatial_size, spatial_size)
                        else:
                            # Fallback: assume square
                            spatial_size = int(np.sqrt(seq_len))
                            cam_spatial = cam[:spatial_size*spatial_size].reshape(spatial_size, spatial_size)
                        
                        print(f"Reshaped CAM to spatial: {cam_spatial.shape}")
                        
                    elif len(self.gradients.shape) == 4:  # [batch, height, width, hidden_dim]
                        weights = torch.mean(self.gradients, dim=(0, 1, 2))  # [hidden_dim]
                        activations = self.activations[0]  # [height, width, hidden_dim]
                        cam_spatial = torch.sum(activations * weights, dim=2)  # [height, width]
                    
                    else:
                        print(f"❌ Unexpected gradient shape: {self.gradients.shape}")
                        return None
                    
                    # Apply ReLU and convert to numpy
                    cam_spatial = torch.relu(cam_spatial)
                    cam = cam_spatial.cpu().numpy()
                    
                    # Resize to image size
                    import cv2
                    cam = cv2.resize(cam, image_size)
                    
                    # Normalize
                    if cam.max() > cam.min():
                        cam = (cam - cam.min()) / (cam.max() - cam.min())
                    
                    print(f"✅ Generated CAM: {cam.shape}, range: [{cam.min():.3f}, {cam.max():.3f}]")
                    return cam
                else:
                    print("❌ No gradients or activations captured")
                    return None
                    
        except Exception as e:
            print(f"❌ Error generating CAM: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description='Simple Grad-CAM test with BLIP')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--question', type=str, required=True, help='Question to ask')
    parser.add_argument('--layer', type=str, default='vision_model.encoder.layers.11', help='Target layer')
    args = parser.parse_args()
    
    # Setup
    config = Config('configs/config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Target layer: {args.layer}")
    
    # Load model
    blip_model = BLIP2VQA(config, train_mode=False)
    blip_model.device = device
    
    model_path = 'checkpoints/blip/checkpoints/best_hf_model'
    if os.path.isdir(model_path):
        blip_model.model = type(blip_model.model).from_pretrained(model_path)
        blip_model.model.to(device)
    
    blip_model.model.eval()
    
    # Load image
    image = Image.open(args.image).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Test normal prediction first
    answer = blip_model.predict(image, args.question)
    print(f"BLIP answer: {answer}")
    
    # Prepare inputs
    inputs = blip_model.processor(image, args.question, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
    
    # Test different layers
    layers_to_test = [
        'vision_model.encoder.layers.11',
        'vision_model.encoder.layers.10', 
        'vision_model.encoder.layers.9',
        'vision_model.post_layernorm'
    ]
    
    for layer_name in layers_to_test:
        print(f"\n=== Testing {layer_name} ===")
        
        grad_cam = SimpleGradCAM(blip_model.model, layer_name)
        
        if grad_cam.register_hooks():
            print("Testing CAM generation...")
            cam = grad_cam.generate_cam_from_vision(inputs, image.size)
            
            if cam is not None:
                # Visualize
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(image)
                axes[0].set_title("Original")
                axes[0].axis('off')
                
                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title(f"CAM - {layer_name}")
                axes[1].axis('off')
                
                # Overlay
                import cv2
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                overlay = cv2.addWeighted(
                    np.array(image), 0.7,
                    heatmap_colored, 0.3,
                    0
                )
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis('off')
                
                plt.suptitle(f"Q: {args.question}\nA: {answer}")
                plt.tight_layout()
                
                output_file = f"gradcam_test_{layer_name.replace('.', '_')}.png"
                plt.savefig(output_file)
                plt.close()
                print(f"✅ Saved result to {output_file}")
                
                # If successful, try this layer in the main GradCAM
                print(f"✅ Layer {layer_name} works! Use this for main implementation.")
                break
            else:
                print(f"❌ CAM generation failed for {layer_name}")
        
        grad_cam.remove_hooks()
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    main()
