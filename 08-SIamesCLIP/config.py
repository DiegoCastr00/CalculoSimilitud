import os
from pathlib import Path
import torch

class Config:
    # Rutas
    DATA_ROOT = Path("./")  # Ajusta esto a tu ruta real
    CHECKPOINT_DIR = Path("checkpoints")
    LOGS_DIR = Path("logs")
    
    # Parámetros de procesamiento de datos
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    IMAGE_SIZE = 224  # Tamaño recomendado para CLIP
    
    # Parámetros del modelo
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM = 512
    TRANSFORMER_HIDDEN_DIM = 1024  # Dimensión oculta de la capa transformadora
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 8
    DROPOUT = 0.1
    
    # Parámetros de entrenamiento
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    TEMPERATURE = 0.07  # Parámetro tau para la pérdida contrastiva
    
    # Dispositivo
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MULTI_GPU = torch.cuda.device_count() > 1
    
    # Configuración para múltiples GPUs
    GPU_IDS = [0, 1]  # Para tus dos RTX 3090
    
    # Ruta al archivo CSV con los datos
    DATA_CSV = DATA_ROOT / "data.csv"
    
    # Frecuencia de guardar checkpoints
    SAVE_EVERY = 5
    
    # Parámetros de validación
    VAL_SPLIT = 0.2  # 20% para validación
    
    # Configuración para text embeddings
    USE_TEXT_EMBEDDINGS = True  # Si usaremos las descripciones de BLIP2
    
    @classmethod
    def create_dirs(cls):
        """Crea los directorios necesarios para el proyecto."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)