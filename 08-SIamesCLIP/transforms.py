import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
from transformers import CLIPFeatureExtractor
from config import Config

class ArtworkSimilarityDataset(Dataset):
    def __init__(self, csv_file, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Ruta al archivo CSV con los datos.
            transform: Transformaciones a aplicar a las imágenes.
            mode (string): 'train' o 'val' para dividir los datos.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(Config.CLIP_MODEL_NAME)
        
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
        
        # Cargar imágenes
        original_image = Image.open(original_img_path).convert('RGB')
        generated_image = Image.open(generated_img_path).convert('RGB')
        negative_image = Image.open(negative_img_path).convert('RGB')
        
        # Procesar imágenes con CLIP
        original_inputs = self.feature_extractor(images=original_image, return_tensors="pt")
        generated_inputs = self.feature_extractor(images=generated_image, return_tensors="pt")
        negative_inputs = self.feature_extractor(images=negative_image, return_tensors="pt")
        
        # Extraer y eliminar la dimensión extra del batch
        original_pixel_values = original_inputs['pixel_values'].squeeze(0)
        generated_pixel_values = generated_inputs['pixel_values'].squeeze(0)
        negative_pixel_values = negative_inputs['pixel_values'].squeeze(0)
        
        # Cargar descripciones si están habilitadas
        if Config.USE_TEXT_EMBEDDINGS:
            original_desc = self.data.iloc[idx]['description_original_paint']
            generated_desc = self.data.iloc[idx]['description_generated_image']
            negative_desc = self.data.iloc[idx]['description_negative_image']
            
            return {
                'original_image': original_pixel_values,
                'generated_image': generated_pixel_values,
                'negative_image': negative_pixel_values,
                'original_desc': original_desc,
                'generated_desc': generated_desc,
                'negative_desc': negative_desc,
            }
        else:
            return {
                'original_image': original_pixel_values,
                'generated_image': generated_pixel_values,
                'negative_image': negative_pixel_values
            }

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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader