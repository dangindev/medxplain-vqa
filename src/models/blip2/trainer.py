import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import time
import numpy as np
from pathlib import Path
import json
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.models.blip2.model import BLIP2VQA
from src.models.blip2.dataset import get_dataloader

logger = logging.getLogger(__name__)

class BLIPTrainer:
    """
    Trainer cho việc fine-tune mô hình BLIP
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tạo thư mục lưu checkpoint
        self.checkpoint_dir = Path(config['model']['blip2']['cache_dir']) / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tạo thư mục lưu logs
        self.logs_dir = Path(config['logging']['save_dir']) / "training"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Khởi tạo mô hình
        logger.info("Initializing model for training...")
        self.model = BLIP2VQA(config, train_mode=True)
        logger.info(f"Model initialized with {self.model.num_parameters:,} parameters")
        
        # Thiết lập training
        self.setup_training()
    
    def setup_training(self):
        """Thiết lập optimizer, scheduler và các thông số cho training"""
        # Tạo dataloaders
        logger.info("Creating dataloaders...")
        self.train_loader, self.train_dataset = get_dataloader(
            self.config, 
            self.model.processor, 
            split='train'
        )
        
        self.val_loader, self.val_dataset = get_dataloader(
            self.config, 
            self.model.processor, 
            split='val'
        )
        
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation dataset: {len(self.val_dataset)} samples")
        
        # Tạo optimizer
        trainable_params = [p for p in self.model.model.parameters() if p.requires_grad]
        logger.info(f"Training with {sum(p.numel() for p in trainable_params):,} trainable parameters")
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Tạo learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs']
        warmup_steps = int(num_training_steps * self.config['training']['warmup_ratio'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training logs
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['fp16'] else None
        
        # BLEU smoother
        self.smoother = SmoothingFunction().method1
    
    def calculate_bleu_score(self, predictions, references):
        """Tính điểm BLEU giữa predictions và references"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            
            if len(pred_tokens) == 0:
                scores.append(0.0)
                continue
            
            try:
                # Tính BLEU score
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoother)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error calculating BLEU score: {e}")
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def train(self):
        """Thực hiện quá trình fine-tuning"""
        num_epochs = self.config['training']['num_epochs']
        log_interval = self.config['logging']['log_interval']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch, log_interval)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Lưu checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch)
            
            # Lưu logs
            self.save_logs()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, "
                      f"Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        
        logger.info("Training completed")
    
    def train_epoch(self, epoch, log_interval):
        """Huấn luyện một epoch"""
        self.model.model.train()
        total_loss = 0
        total_bleu = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass với mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model.forward(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values'],
                        labels=batch['labels']
                    )
                    
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        logger.error("Model output does not have 'loss' attribute")
                        continue
                
                # Backward và optimize
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['max_grad_norm'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                outputs = self.model.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logger.error("Model output does not have 'loss' attribute")
                    continue
                
                # Backward và optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                
                self.optimizer.step()
            
            # Cập nhật scheduler
            self.scheduler.step()
            
            # Generate predictions để tính BLEU score
            with torch.no_grad():
                generated_answers = self.model.generate_answers(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Tính BLEU score
                bleu_score = self.calculate_bleu_score(
                    generated_answers, 
                    batch['answer_text']
                )
                total_bleu += bleu_score
            
            # Cập nhật loss
            total_loss += loss.item()
            num_batches += 1
            
            # Cập nhật progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bleu': f"{bleu_score:.4f}"
            })
            
            # Log
            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] - "
                          f"Loss: {loss.item():.4f}, BLEU: {bleu_score:.4f}")
        
        pbar.close()
        
        # Tính toán loss và accuracy trung bình
        avg_loss = total_loss / max(1, num_batches)
        avg_bleu = total_bleu / max(1, num_batches)
        
        return avg_loss, avg_bleu
    
    def validate_epoch(self, epoch):
        """Đánh giá mô hình trên tập validation"""
        self.model.model.eval()
        total_loss = 0
        total_bleu = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Forward pass
                outputs = self.model.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logger.error("Model output does not have 'loss' attribute during validation")
                    continue
                
                # Generate predictions để tính BLEU score
                generated_answers = self.model.generate_answers(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Tính BLEU score
                bleu_score = self.calculate_bleu_score(
                    generated_answers, 
                    batch['answer_text']
                )
                total_bleu += bleu_score
                
                # Cập nhật loss
                total_loss += loss.item()
                num_batches += 1
                
                # Cập nhật progress bar
                pbar.update(1)
        
        pbar.close()
        
        # Tính toán loss và accuracy trung bình
        avg_loss = total_loss / max(1, num_batches)
        avg_bleu = total_bleu / max(1, num_batches)
        
        return avg_loss, avg_bleu
    
    def save_checkpoint(self, epoch, is_best=False):
        """Lưu checkpoint của mô hình"""
        # Checkpoint path
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        # Tạo state dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        # Lưu checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Đồng thời lưu mô hình HuggingFace
        if is_best:
            hf_path = self.checkpoint_dir / "best_hf_model"
            self.model.save_pretrained(hf_path)
    
    def save_logs(self):
        """Lưu logs huấn luyện"""
        logs = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        # Convert numpy arrays to lists
        for k, v in logs.items():
            if isinstance(v, np.ndarray):
                logs[k] = v.tolist()
        
        # Lưu logs
        logs_path = self.logs_dir / "training_logs.json"
        with open(logs_path, 'w') as f:
            json.dump(logs, f)
    
    def load_checkpoint(self, checkpoint_path):
        """Tải checkpoint cho mô hình"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint {checkpoint_path} not found")
            return False
        
        # Tải checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state dict nếu đang ở chế độ training
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state dict nếu có
        if hasattr(self, 'scheduler') and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        if 'train_accuracies' in checkpoint:
            self.train_accuracies = checkpoint['train_accuracies']
        
        if 'val_accuracies' in checkpoint:
            self.val_accuracies = checkpoint['val_accuracies']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return True
