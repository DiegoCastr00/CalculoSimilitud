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
        original_img_path = os.path.join(Config.IMAGES_DIR, self.data.iloc[idx]['original_image'])
        generated_img_path = os.path.join(Config.IMAGES_DIR, self.data.iloc[idx]['generated_image'])
        negative_img_path = os.path.join(Config.IMAGES_DIR, self.data.iloc[idx]['negative_image'])
        
        # Control de errores con contador para evitar bucles infinitos
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                original_image = Image.open(original_img_path).convert('RGB')
                generated_image = Image.open(generated_img_path).convert('RGB')
                negative_image = Image.open(negative_img_path).convert('RGB')
                break
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                retry_count += 1
                idx = np.random.randint(0, len(self.data))
                original_img_path = os.path.join(Config.IMAGES_DIR, self.data.iloc[idx]['original_image'])
                generated_img_path = os.path.join(Config.IMAGES_DIR, self.data.iloc[idx]['generated_image'])
                negative_img_path = os.path.join(Config.IMAGES_DIR, self.data.iloc[idx]['negative_image'])
                
                if retry_count == max_retries:
                    # Si falla después de varios intentos, crear imágenes en blanco
                    print("Maximum retries reached. Using blank images.")
                    original_image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                    generated_image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                    negative_image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        
        # Inicializar muestra
        sample = {}
        
        # Obtener descripciones
        if Config.USE_TEXT_EMBEDDINGS:
            original_desc = self.data.iloc[idx].get('description_original_paint', '')
            generated_desc = self.data.iloc[idx].get('description_generated_image', '')
            negative_desc = self.data.iloc[idx].get('description_negative_image', '')
            
            # Evitar descripciones vacías o NaN
            original_desc = original_desc if pd.notna(original_desc) and original_desc != '' else "A piece of artwork"
            generated_desc = generated_desc if pd.notna(generated_desc) and generated_desc != '' else "A piece of artwork"
            negative_desc = negative_desc if pd.notna(negative_desc) and negative_desc != '' else "A piece of artwork"
            
            # Procesar cada par (imagen, texto) por separado
            # Original
            original_inputs = self.processor(
                text=original_desc,
                images=original_image, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True,
                max_length=77  # Longitud máxima para CLIP
            )
            
            # Generated
            generated_inputs = self.processor(
                text=generated_desc,
                images=generated_image, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True,
                max_length=77
            )
            
            # Negative
            negative_inputs = self.processor(
                text=negative_desc,
                images=negative_image, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True,
                max_length=77
            )
            
            # Extraer valores para muestra
            for prefix, inputs in zip(
                ['original', 'generated', 'negative'],
                [original_inputs, generated_inputs, negative_inputs]
            ):
                sample[f'{prefix}_pixel_values'] = inputs.pixel_values.squeeze(0)
                sample[f'{prefix}_input_ids'] = inputs.input_ids.squeeze(0)
                sample[f'{prefix}_attention_mask'] = inputs.attention_mask.squeeze(0)
        else:
            # Solo procesar imágenes sin texto
            for prefix, image in zip(
                ['original', 'generated', 'negative'],
                [original_image, generated_image, negative_image]
            ):
                inputs = self.processor(images=image, return_tensors="pt")
                sample[f'{prefix}_pixel_values'] = inputs.pixel_values.squeeze(0)
        
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