import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from transformers import CLIPFeatureExtractor
from typing import Tuple, Dict, List, Union, Optional
from config import Config

class ArtSimilarityDataset(Dataset):
    """
    Dataset para entrenar un modelo de similitud de imágenes artísticas.
    Carga tripletes de imágenes: original, generada por IA, y negativa (diferente).
    """
    def __init__(
        self, 
        csv_path: str, 
        mode: str = 'train',
        train_ratio: float = 0.9,
        transform = None,
        seed: int = 42
    ):
        """
        Args:
            csv_path: Ruta al archivo CSV con los datos
            mode: 'train' o 'val' para dividir los datos
            train_ratio: Proporción de datos para entrenamiento
            transform: Transformaciones a aplicar a las imágenes
            seed: Semilla para reproducibilidad
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform or CLIPFeatureExtractor.from_pretrained(
            Config.CLIP_MODEL_NAME
        )
        
        # Dividir en entrenamiento y validación
        random.seed(seed)
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:  # val
            self.indices = indices[split_idx:]
            
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        idx = self.indices[idx]
        row = self.data.iloc[idx]
        
        # Cargar las tres imágenes
        original_img = self._load_and_preprocess_image(row['original_image'])
        generated_img = self._load_and_preprocess_image(row['generated_image'])
        negative_img = self._load_and_preprocess_image(row['negative_image'])
        
        # Cargar las descripciones
        original_desc = row['description_original_paint']
        generated_desc = row['description_generated_image']
        negative_desc = row['description_negative_image']
        
        return {
            'original_image': original_img,
            'generated_image': generated_img,
            'negative_image': negative_img,
            'original_desc': original_desc,
            'generated_desc': generated_desc,
            'negative_desc': negative_desc
        }
    
    def _load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        """Carga y preprocesa una imagen utilizando el transformador de CLIP."""
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Aplicar transformaciones de CLIP
            processed = self.transform(images=img, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            # En caso de error, crear un tensor de ceros
            return torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))


def create_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """
    Crea los dataloaders para entrenamiento y validación.
    
    Returns:
        train_loader: DataLoader para el conjunto de entrenamiento
        val_loader: DataLoader para el conjunto de validación
    """
    # Configurar el dataset
    train_dataset = ArtSimilarityDataset(
        csv_path=Config.DATA_CSV,
        mode='train',
        train_ratio=1 - Config.VALIDATION_SPLIT
    )
    
    val_dataset = ArtSimilarityDataset(
        csv_path=Config.DATA_CSV,
        mode='val',
        train_ratio=1 - Config.VALIDATION_SPLIT
    )
    
    # Configurar los dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader