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
    # Establecer semilla para reproducibilidad
    seed = 42
    set_seed(seed)
    
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