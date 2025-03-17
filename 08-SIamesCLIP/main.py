import argparse
import torch
import wandb
import os
import random
import numpy as np
from datetime import datetime

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

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de similitud artística")
    parser.add_argument("--data_root", type=str, default=None, help="Ruta a los datos")
    parser.add_argument("--batch_size", type=int, default=None, help="Tamaño del batch")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=None, help="Tasa de aprendizaje")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--use_wandb", action="store_true", help="Usar Weights & Biases para seguimiento")
    parser.add_argument("--project_name", type=str, default="art-similarity", help="Nombre del proyecto en W&B")
    parser.add_argument("--use_text", action="store_true", help="Usar embeddings de texto")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="IDs de GPU a utilizar (separados por comas)")
    
    args = parser.parse_args()
    
    # Establecer semilla
    set_seed(args.seed)
    
    # Actualizar configuración con argumentos CLI
    if args.data_root:
        Config.DATA_ROOT = args.data_root
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.use_text is not None:
        Config.USE_TEXT_EMBEDDINGS = args.use_text
    
    # Configurar GPUs
    if args.gpu_ids:
        Config.GPU_IDS = [int(id) for id in args.gpu_ids.split(',')]
    
    # Inicializar W&B
    if args.use_wandb:
        run_name = f"artsiamese-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.project_name,
            name=run_name,
            config={
                "learning_rate": Config.LEARNING_RATE,
                "batch_size": Config.BATCH_SIZE,
                "epochs": Config.EPOCHS,
                "model": Config.CLIP_MODEL_NAME,
                "transformer_layers": Config.TRANSFORMER_LAYERS,
                "use_text": Config.USE_TEXT_EMBEDDINGS,
                "temperature": Config.TEMPERATURE,
            }
        )
    
    # Crear entrenador
    trainer = Trainer(Config)
    
    # Entrenar
    trainer.train()
    
    # Cerrar W&B
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()