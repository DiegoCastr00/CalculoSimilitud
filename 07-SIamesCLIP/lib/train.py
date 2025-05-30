import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
import numpy as np
import time
import os
import logging
import traceback
from tqdm import tqdm
from pathlib import Path
import wandb

from model import SiameseCLIPModel
from loss import MultimodalContrastiveLoss
from transforms import get_dataloaders
from config import Config

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
        
        # Crear directorios necesarios
        config.create_dirs()
        
        # Inicializar modelo
        self.model = SiameseCLIPModel(config)
        self.model.to(self.device)
        
        # Paralelizar modelo si hay múltiples GPUs disponibles
        if config.MULTI_GPU:
            self.model = nn.DataParallel(self.model, device_ids=config.GPU_IDS)
        
        # Inicializar optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Inicializar scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Inicializar función de pérdida
        self.criterion = MultimodalContrastiveLoss(
            temperature=config.TEMPERATURE,
            alpha_text=0.3,  # Peso para la pérdida de texto
            alpha_cross=0.3  # Peso para la pérdida cruzada
        )
        
        # Inicializar dataloaders
        self.train_loader, self.val_loader = get_dataloaders(config)
        
        # Inicializar grad scaler para mixed precision
        self.scaler = GradScaler('cuda')
        
        # Inicializar métricas
        self.best_val_loss = float('inf')
        
        # Inicializar variables para Early Stopping
        self.early_stopping_enabled = config.EARLY_STOPPING
        self.patience = config.PATIENCE
        self.min_delta = config.MIN_DELTA
        self.counter = 0
        self.early_stop = False
        
    def train_epoch(self, epoch):
        """
        Entrena el modelo por una época.
        """
        self.model.train()
        epoch_loss = 0
        epoch_triplet_acc = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover datos a GPU
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass con mixed precision
            self.optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                # Pasar el batch completo al modelo en lugar de desempaquetar los inputs
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs)
                loss = loss_dict['total_loss']
            
            # Backward pass con mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Actualizar métricas
            epoch_loss += loss.item()
            epoch_triplet_acc += loss_dict['triplet_accuracy']
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss': loss.item(),
                'triplet_acc': loss_dict['triplet_accuracy']
            })
        
        # Calcular métricas promedio
        epoch_loss /= len(self.train_loader)
        epoch_triplet_acc /= len(self.train_loader)
        
        # Registrar métricas en wandb
        if wandb.run is not None:
            wandb.log({
                'train_loss': epoch_loss,
                'train_triplet_acc': epoch_triplet_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
        
        return epoch_loss, epoch_triplet_acc
    
    def validate(self, epoch):
        """
        Valida el modelo en el conjunto de validación.
        """
        self.model.eval()
        val_loss = 0
        val_triplet_acc = 0
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Mover datos a GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Pasar el batch completo al modelo en lugar de desempaquetar los inputs
                # Forward pass
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs)
                
                # Actualizar métricas
                val_loss += loss_dict['total_loss'].item()
                val_triplet_acc += loss_dict['triplet_accuracy']
                
                # Acumular otras métricas
                for key, value in loss_dict.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0
                    if isinstance(value, torch.Tensor):
                        val_metrics[key] += value.item()
                    else:
                        val_metrics[key] += value
        
        # Calcular métricas promedio
        val_loss /= len(self.val_loader)
        val_triplet_acc /= len(self.val_loader)
        
        for key in val_metrics:
            val_metrics[key] /= len(self.val_loader)
        
        # Registrar métricas en wandb
        if wandb.run is not None:
            wandb.log({
                'val_loss': val_loss,
                'val_triplet_acc': val_triplet_acc,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'epoch': epoch
            })
        
        # Actualizar scheduler
        self.scheduler.step(val_loss)
        
        # Lógica de Early Stopping
        if self.early_stopping_enabled:
            if val_loss < (self.best_val_loss - self.min_delta):
                # Si hay una mejora significativa, resetear el contador
                self.best_val_loss = val_loss
                self.counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logging.info(f"Nueva mejor pérdida de validación: {val_loss:.4f}. Guardando mejor modelo.")
            else:
                # Si no hay mejora significativa, incrementar el contador
                self.counter += 1
                if self.counter >= self.patience:
                    logging.info(f"Early stopping triggered after {epoch+1} épocas. No hay mejora en la pérdida de validación durante {self.patience} épocas.")
                    self.early_stop = True
                logging.info(f"Early stopping counter: {self.counter}/{self.patience}")
        elif val_loss < self.best_val_loss:
            # Si Early Stopping está desactivado, seguir con el comportamiento normal
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best=True)
            logging.info(f"Nueva mejor pérdida de validación: {val_loss:.4f}. Guardando mejor modelo.")
            
        return val_loss, val_triplet_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Guarda un checkpoint del modelo.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if not self.config.MULTI_GPU else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scaler': self.scaler.state_dict()  # Guardar estado del scaler
        }
        
        # Guardar checkpoint regular
        if (epoch + 1) % self.config.SAVE_EVERY == 0:
            torch.save(
                checkpoint,
                os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
            )
        
        # Guardar mejor modelo
        if is_best:
            torch.save(
                checkpoint,
                os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
            )
    
    def train(self):
        """
        Entrena el modelo por el número de épocas especificado.
        """
        try:
            logging.info(f"Training on {self.device}")
            logging.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
            logging.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
            logging.info(f"Usando {torch.cuda.device_count()} GPUs: {self.config.GPU_IDS}")
            
            for epoch in range(self.config.EPOCHS):
                start_time = time.time()
                
                # Entrenar época
                train_loss, train_triplet_acc = self.train_epoch(epoch)
                
                # Validar
                val_loss, val_triplet_acc = self.validate(epoch)
                
                # Calcular tiempo
                epoch_time = time.time() - start_time
                
                # Registrar métricas
                log_msg = (f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Train Triplet Acc: {train_triplet_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Triplet Acc: {val_triplet_acc:.4f} | "
                          f"Time: {epoch_time:.2f}s")
                logging.info(log_msg)
                
                # Guardar checkpoint regular
                if (epoch + 1) % self.config.SAVE_EVERY == 0:
                    self.save_checkpoint(epoch)
                    logging.info(f"Guardado checkpoint de la época {epoch+1}")
                
                # Verificar si debemos detener el entrenamiento por Early Stopping
                if self.early_stop:
                    logging.info(f"Entrenamiento detenido tempranamente en la época {epoch+1}")
                    break
            
            logging.info("Entrenamiento completado con éxito!")
        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {str(e)}")
            logging.error(traceback.format_exc())
            raise
