import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from config import Config
from typing import Tuple, Dict, List, Union, Optional

class TransformerEncoder(nn.Module):
    """
    Módulo de encoder transformer para procesar embeddings de CLIP.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.linear_out = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, embedding_dim)
        # Añadir dimensión de secuencia para transformer
        x = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Proyectar a dimensión oculta
        x = self.linear_in(x)
        
        # Pasar por transformer
        x = self.transformer(x)
        
        # Proyectar de vuelta a dimensión original
        x = self.linear_out(x)
        
        # Eliminar dimensión de secuencia
        x = x.squeeze(1)  # (batch_size, embedding_dim)
        
        # Normalizar y aplicar dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class SiameseImageSimilarityModel(nn.Module):
    """
    Modelo siamés para calcular similitudes entre imágenes artísticas.
    Utiliza CLIP como extractor de características congelado y añade
    capas transformadoras para refinar los embeddings.
    """
    def __init__(
        self,
        clip_model_name: str = Config.CLIP_MODEL_NAME,
        embedding_dim: int = Config.EMBEDDING_DIM,
        transformer_hidden_dim: int = Config.TRANSFORMER_HIDDEN_DIM,
        transformer_layers: int = Config.TRANSFORMER_LAYERS,
        transformer_heads: int = Config.TRANSFORMER_HEADS,
        dropout: float = Config.DROPOUT
    ):
        super().__init__()
        
        # Cargar modelo CLIP preentrenado
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Congelar CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Dimensión del embedding de CLIP para imágenes
        self.clip_embedding_dim = self.clip.vision_model.config.hidden_size
        
        # Añadir capas de transformador para refinar embeddings
        self.transformer = TransformerEncoder(
            input_dim=self.clip_embedding_dim,
            hidden_dim=transformer_hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout
        )
        
        # Proyector final para normalizar embeddings a la dimensión deseada
        if embedding_dim != self.clip_embedding_dim:
            self.projector = nn.Linear(self.clip_embedding_dim, embedding_dim)
        else:
            self.projector = nn.Identity()
            
    def get_clip_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extrae embeddings de imágenes usando CLIP.
        
        Args:
            images: Tensor de imágenes (batch_size, 3, H, W)
            
        Returns:
            Embeddings de CLIP (batch_size, clip_embedding_dim)
        """
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(
                pixel_values=images,
                output_hidden_states=False
            )
            image_embeds = vision_outputs[1]  # Obtiene el embedding de [CLS]
            
        return image_embeds
    
    def forward_one(self, image: torch.Tensor) -> torch.Tensor:
        """
        Proceso una sola imagen a través del pipeline completo.
        
        Args:
            image: Tensor de imagen (batch_size, 3, H, W)
            
        Returns:
            Embedding final refinado y normalizado (batch_size, embedding_dim)
        """
        # Extraer embeddings de CLIP
        clip_embedding = self.get_clip_image_embeddings(image)
        
        # Refinar con transformer
        refined_embedding = self.transformer(clip_embedding)
        
        # Proyectar a la dimensión deseada
        projected_embedding = self.projector(refined_embedding)
        
        # Normalizar el embedding para calcular similitud por coseno
        normalized_embedding = F.normalize(projected_embedding, p=2, dim=1)
        
        return normalized_embedding
    
    def forward(
        self, 
        original_images: torch.Tensor, 
        generated_images: torch.Tensor, 
        negative_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Procesa tripletes de imágenes (original, generada, negativa).
        
        Args:
            original_images: Tensor de imágenes originales (batch_size, 3, H, W)
            generated_images: Tensor de imágenes generadas (batch_size, 3, H, W)
            negative_images: Tensor de imágenes negativas (batch_size, 3, H, W)
            
        Returns:
            Embeddings normalizados para cada tipo de imagen
        """
        original_embeddings = self.forward_one(original_images)
        generated_embeddings = self.forward_one(generated_images)
        negative_embeddings = self.forward_one(negative_images)
        
        return original_embeddings, generated_embeddings, negative_embeddings
    
    def compute_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calcula la similitud coseno entre dos imágenes.
        
        Args:
            img1: Primera imagen (batch_size, 3, H, W)
            img2: Segunda imagen (batch_size, 3, H, W)
            
        Returns:
            Similitud coseno entre las imágenes (batch_size)
        """
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        
        # Calcular similitud coseno
        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        
        return similarity