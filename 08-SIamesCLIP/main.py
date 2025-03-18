import torch
import wandb
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

from config import Config
from train import Trainer

def set_seed(seed):
    """
    Establece semillas aleatorias para reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_data_structure():
    """
    Verifica que la estructura de carpetas y el CSV sean correctos.
    Comprueba que el archivo CSV existe y que al menos la primera imagen de cada tipo es accesible.
    """
    print("\nVerificando estructura de datos...")
    
    # Verificar que existe el archivo CSV
    csv_path = Config.DATA_CSV
    if not csv_path.exists():
        print(f"ERROR: El archivo CSV no existe en la ruta: {csv_path}")
        return False
    
    # Verificar que el CSV tiene contenido
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"ERROR: El archivo CSV está vacío: {csv_path}")
            return False
        
        print(f"CSV cargado correctamente con {len(df)} filas.")
        
        # Verificar que el CSV tiene las columnas necesarias
        required_columns = ['original_image', 'generated_image', 'negative_image']
        if Config.USE_TEXT_EMBEDDINGS:
            required_columns.extend(['description_original_paint', 'description_generated_image', 'description_negative_image'])
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Faltan las siguientes columnas en el CSV: {missing_columns}")
            return False
        
        print("Todas las columnas requeridas están presentes en el CSV.")
        
        # Verificar que existe la carpeta de imágenes
        if not Config.IMAGES_DIR.exists():
            print(f"ERROR: La carpeta de imágenes no existe: {Config.IMAGES_DIR}")
            return False
        
        # Verificar que se puede acceder a la primera imagen de cada tipo
        first_row = df.iloc[0]
        image_paths = [
            os.path.join(Config.IMAGES_DIR, first_row['original_image']),
            os.path.join(Config.IMAGES_DIR, first_row['generated_image']),
            os.path.join(Config.IMAGES_DIR, first_row['negative_image'])
        ]
        
        for i, path in enumerate(['original', 'generated', 'negative']):
            if not os.path.exists(image_paths[i]):
                print(f"ERROR: No se puede acceder a la imagen {path}: {image_paths[i]}")
                return False
        
        print("Se ha verificado el acceso a las primeras imágenes correctamente.")
        return True
        
    except Exception as e:
        print(f"ERROR al procesar el CSV: {e}")
        return False

def main():
    # Establecer semilla para reproducibilidad
    seed = 42
    set_seed(seed)
    
    # Crear directorios necesarios
    Config.create_dirs()
    
    # Verificar estructura de datos antes de iniciar el entrenamiento
    if not verify_data_structure():
        print("\nERROR: La verificación de datos ha fallado. Por favor, corrige los problemas antes de continuar.")
        sys.exit(1)
    
    print("\nVerificación de datos completada con éxito. Iniciando entrenamiento...\n")
    
    # Configurar W&B para seguimiento de métricas
    use_wandb = True  # Cambiar a False si no se desea usar W&B
    
    if use_wandb:
        run_name = f"artsiamese-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project="art-similarity",
            name=run_name,
            config={
                "learning_rate": Config.LEARNING_RATE,
                "batch_size": Config.BATCH_SIZE,
                "epochs": Config.EPOCHS,
                "model": Config.CLIP_MODEL_NAME,
                "transformer_layers": Config.TRANSFORMER_LAYERS,
                "use_text": Config.USE_TEXT_EMBEDDINGS,
                "temperature": Config.TEMPERATURE,
                "early_stopping": Config.EARLY_STOPPING,
                "patience": Config.PATIENCE,
                "min_delta": Config.MIN_DELTA,
            }
        )
    
    # Crear entrenador
    trainer = Trainer(Config)
    
    # Entrenar
    trainer.train()
    
    # Cerrar W&B
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()