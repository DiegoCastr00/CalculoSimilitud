import torch
import wandb
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import logging
import traceback

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
    # Verificar que existe el archivo CSV
    csv_path = Config.DATA_CSV
    if not csv_path.exists():
        logging.error(f"El archivo CSV no existe en la ruta: {csv_path}")
        return False
    
    # Verificar que el CSV tiene contenido
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            logging.error(f"El archivo CSV está vacío: {csv_path}")
            return False
        
        logging.info(f"CSV cargado correctamente con {len(df)} filas.")
        
        # Verificar que el CSV tiene las columnas necesarias
        required_columns = ['original_image', 'generated_image', 'negative_image']
        if Config.USE_TEXT_EMBEDDINGS:
            required_columns.extend(['description_original_image', 'description_generated_image', 'description_negative_image'])
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Faltan las siguientes columnas en el CSV: {missing_columns}")
            return False
        
        logging.info("Todas las columnas requeridas están presentes en el CSV.")
        
        # Verificar que existe la carpeta de imágenes
        if not Config.IMAGES_DIR.exists():
            logging.error(f"La carpeta de imágenes no existe: {Config.IMAGES_DIR}")
            return False
        
        # Verificar que se puede acceder a la primera imagen de cada tipo
        first_row = df.iloc[0]
        image_paths = [
            os.path.join(Config.IMAGES_DIR, first_row['original_image']),
            os.path.join(Config.IMAGES_DIR, first_row['generated_image']),
            os.path.join(Config.IMAGES_DIR, first_row['negative_image'])
        ]
        
        for i, path_type in enumerate(['original', 'generated', 'negative']):
            if not os.path.exists(image_paths[i]):
                logging.error(f"No se puede acceder a la imagen {path_type}: {image_paths[i]}")
                return False
        
        logging.info("Se ha verificado el acceso a las primeras imágenes correctamente.")
        return True
        
    except Exception as e:
        logging.error(f"ERROR al procesar el CSV: {e}")
        logging.error(traceback.format_exc())
        return False

def setup_logging():
    """Configura el sistema de logging para capturar la salida cuando se ejecute con nohup"""
    # Crear directorio de logs si no existe
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    # Redireccionar excepciones no capturadas al log
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # No capturar interrupciones de teclado
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logging.error("Excepción no capturada:", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    logging.info(f"Iniciando entrenamiento con timestamp: {Config.RUN_TIMESTAMP}")
    logging.info(f"Logs guardados en: {Config.LOG_FILE}")
    logging.info(f"Checkpoints guardados en: {Config.CHECKPOINT_DIR}")

def main():
    try:
        # Establecer semilla para reproducibilidad
        seed = 42
        set_seed(seed)
        
        # Crear directorios necesarios
        Config.create_dirs()
        
        # Configurar logging
        setup_logging()
        
        # Verificar estructura de datos antes de iniciar el entrenamiento
        logging.info("Verificando estructura de datos...")
        if not verify_data_structure():
            logging.error("La verificación de datos ha fallado. Por favor, corrige los problemas antes de continuar.")
            sys.exit(1)
        
        logging.info("Verificación de datos completada con éxito. Iniciando entrenamiento...")
    
        # Configurar W&B para seguimiento de métricas
        use_wandb = True  # Cambiar a False si no se desea usar W&B
        
        if use_wandb:
            run_name = f"artsiamese-{Config.RUN_TIMESTAMP}"
            logging.info(f"Iniciando W&B run: {run_name}")
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
        logging.info("Creando entrenador...")
        trainer = Trainer(Config)
        
        # Entrenar
        logging.info("Iniciando entrenamiento...")
        trainer.train()
        
        # Cerrar W&B
        if use_wandb:
            logging.info("Finalizando W&B run...")
            wandb.finish()
            
        logging.info("Entrenamiento completado con éxito!")
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        logging.error(traceback.format_exc())
        if 'wandb' in locals() and use_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)

if __name__ == "__main__":
    main()