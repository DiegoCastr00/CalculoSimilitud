import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
from transformers import CLIPProcessor
from config import Config

class ArtworkSimilarityDataset(Dataset):
    def __init__(self, csv_file, mode='train'):
        """
        Args:
            csv_file (string): Ruta al archivo CSV con los datos.
            mode (string): 'train' o 'val' para dividir los datos.
        """
        self.data = pd.read_csv(csv_file)
        # Usamos CLIPProcessor que maneja tanto imágenes como texto
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
        
        # Dividir en train/val si es necesario
        if mode == 'train':
            self.data = self.data.iloc[:int(len(self.data) * (1 - Config.VAL_SPLIT))]
        elif mode == 'val':
            self.data = self.data.iloc[int(len(self.data) * (1 - Config.VAL_SPLIT)):]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Obtener rutas de imágenes
        original_img_path = os.path.join(Config.DATA_ROOT, self.data.iloc[idx]['original_image'])
        generated_img_path = os.path.join(Config.DATA_ROOT, self.data.iloc[idx]['generated_image'])
        negative_img_path = os.path.join(Config.DATA_ROOT, self.data.iloc[idx]['negative_image'])
        
        # Manejo de errores para imágenes
        try:
            original_image = Image.open(original_img_path).convert('RGB')
            generated_image = Image.open(generated_img_path).convert('RGB')
            negative_image = Image.open(negative_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Usar índice aleatorio diferente como fallback
            return self.__getitem__(np.random.randint(0, len(self.data)))
        
        # Obtener descripciones
        sample = {}
        
        if Config.USE_TEXT_EMBEDDINGS:
            original_desc = self.data.iloc[idx].get('description_original_paint', '')
            generated_desc = self.data.iloc[idx].get('description_generated_image', '')
            negative_desc = self.data.iloc[idx].get('description_negative_image', '')
            
            # Evitar descripciones vacías
            original_desc = original_desc if pd.notna(original_desc) else "A piece of artwork"
            generated_desc = generated_desc if pd.notna(generated_desc) else "A piece of artwork"
            negative_desc = negative_desc if pd.notna(negative_desc) else "A piece of artwork"
            
            # Procesar imagen+texto para CLIP
            original_inputs = self.processor(
                text=original_desc,
                images=original_image, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            generated_inputs = self.processor(
                text=generated_desc,
                images=generated_image, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            negative_inputs = self.processor(
                text=negative_desc,
                images=negative_image, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # Extraer valores
            sample['original_pixel_values'] = original_inputs['pixel_values'].squeeze(0)
            sample['generated_pixel_values'] = generated_inputs['pixel_values'].squeeze(0)
            sample['negative_pixel_values'] = negative_inputs['pixel_values'].squeeze(0)
            
            sample['original_input_ids'] = original_inputs['input_ids'].squeeze(0)
            sample['generated_input_ids'] = generated_inputs['input_ids'].squeeze(0)
            sample['negative_input_ids'] = negative_inputs['input_ids'].squeeze(0)
            
            sample['original_attention_mask'] = original_inputs.get('attention_mask', torch.ones_like(original_inputs['input_ids'])).squeeze(0)
            sample['generated_attention_mask'] = generated_inputs.get('attention_mask', torch.ones_like(generated_inputs['input_ids'])).squeeze(0)
            sample['negative_attention_mask'] = negative_inputs.get('attention_mask', torch.ones_like(negative_inputs['input_ids'])).squeeze(0)
        else:
            # Solo procesar imágenes
            original_inputs = self.processor(images=original_image, return_tensors="pt")
            generated_inputs = self.processor(images=generated_image, return_tensors="pt")
            negative_inputs = self.processor(images=negative_image, return_tensors="pt")
            
            sample['original_pixel_values'] = original_inputs['pixel_values'].squeeze(0)
            sample['generated_pixel_values'] = generated_inputs['pixel_values'].squeeze(0)
            sample['negative_pixel_values'] = negative_inputs['pixel_values'].squeeze(0)
        
        return sample

def get_dataloaders(config):
    """
    Crea los dataloaders para entrenamiento y validación.
    
    Args:
        config: Configuración del proyecto.
    
    Returns:
        train_loader, val_loader
    """
    # Crear datasets
    train_dataset = ArtworkSimilarityDataset(
        csv_file=config.DATA_CSV,
        mode='train'
    )
    
    val_dataset = ArtworkSimilarityDataset(
        csv_file=config.DATA_CSV,
        mode='val'
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # Evita problemas con batches incompletos
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader