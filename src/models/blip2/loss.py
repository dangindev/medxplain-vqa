import torch
import torch.nn as nn
import torch.nn.functional as F

class VQALoss:
    def __init__(self, config):
        self.config = config
        self.loss_type = config['loss']['type']
        self.label_smoothing = config['loss'].get('label_smoothing', 0.0)
        
    def compute_loss(self, outputs, targets):
        """Tính toán loss dựa vào loại được cấu hình"""
        if self.loss_type == "cross_entropy":
            return self._cross_entropy_loss(outputs, targets)
        elif self.loss_type == "focal":
            return self._focal_loss(outputs, targets)
        elif self.loss_type == "contrastive":
            return self._contrastive_loss(outputs, targets)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _cross_entropy_loss(self, outputs, targets):
        """Cross entropy loss với label smoothing"""
        lm_logits = outputs.logits
        
        # Shift để dự đoán token tiếp theo
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        # Tính cross entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            label_smoothing=self.label_smoothing,
            ignore_index=-100  # Ignore padding
        )
        
        return loss
    
    def _focal_loss(self, outputs, targets, gamma=2.0):
        """Focal loss cho imbalanced datasets"""
        lm_logits = outputs.logits
        
        # Shift để dự đoán token tiếp theo
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        # Chuyển đổi shape
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Tính xác suất
        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Lấy xác suất của các nhãn đúng
        pt = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        pt = torch.where(shift_labels != -100, pt, torch.ones_like(pt))
        
        # Tính focal loss
        loss = -((1 - pt) ** gamma) * log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        loss = torch.where(shift_labels != -100, loss, torch.zeros_like(loss))
        
        return loss.mean()
    
    def _contrastive_loss(self, outputs, targets, temperature=0.5):
        """Contrastive loss cho VQA"""
        # Đơn giản hóa: sử dụng cross entropy nhưng có thể triển khai contrastive loss chi tiết hơn sau
        return self._cross_entropy_loss(outputs, targets)
