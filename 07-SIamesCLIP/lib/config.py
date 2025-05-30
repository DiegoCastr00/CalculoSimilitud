import os
from pathlib import Path
import torch
from datetime import datetime

class Config:
    # Rutas
    DATA_ROOT = Path("/home/jesusgr/SimilitudImagenes/08-SiamesCLIP")  # Directorio raíz del proyecto
    IMAGES_DIR = DATA_ROOT / "imagenes"  # Carpeta de imágenes
    
    # Timestamp para identificar la ejecución actual
    RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Directorios para checkpoints y logs con timestamp para evitar sobrescrituras
    CHECKPOINT_DIR = Path(f"checkpoints_{RUN_TIMESTAMP}")
    LOGS_DIR = Path(f"logs_{RUN_TIMESTAMP}")
    LOG_FILE = LOGS_DIR / f"training_{RUN_TIMESTAMP}.log"
    
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
    VAL_SPLIT = 0.6  # 50% para validación
    
    # Parámetros de Early Stopping
    EARLY_STOPPING = True  # Activar/desactivar early stopping
    PATIENCE = 5  # Número de épocas a esperar antes de detener el entrenamiento
    MIN_DELTA = 0.001  # Cambio mínimo en la métrica para considerarla una mejora
    
    # Configuración para text embeddings
    USE_TEXT_EMBEDDINGS = True  # Si usaremos las descripciones de BLIP2
    
    @classmethod
    def create_dirs(cls):
        """Crea los directorios necesarios para el proyecto."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)