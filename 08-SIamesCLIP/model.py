import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from config import Config

class TransformerProjection(nn.Module):
    """
    Capa transformadora para refinar los embeddings de CLIP.
    """
    def __init__(self, embed_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerProjection, self).__init__()
        
        # Capa de normalización inicial
        self.norm = nn.LayerNorm(embed_dim)
        
        # Capas del transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Proyección final
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        # x tiene forma [batch_size, embed_dim]
        # Añadimos una dimensión de secuencia para el transformer
        x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Normalización
        x = self.norm(x)
        
        # Pasar por transformer
        x = self.transformer_encoder(x)
        
        # Eliminar dimensión de secuencia
        x = x.squeeze(1)  # [batch_size, embed_dim]
        
        # Proyección final
        x = self.projection(x)
        
        # Normalizar para similitud del coseno
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SiameseCLIPModel(nn.Module):
    """
    Modelo siamés basado en CLIP con capas transformadoras adicionales.
    """
    def __init__(self, config):
        super(SiameseCLIPModel, self).__init__()
        
        # Cargar modelo CLIP preentrenado
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        
        # Congelar pesos de CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Dimensión del embedding de CLIP
        self.embed_dim = config.EMBEDDING_DIM
        
        # Capas transformadoras para refinamiento
        self.transformer_projection = TransformerProjection(
            embed_dim=self.embed_dim,
            hidden_dim=config.TRANSFORMER_HIDDEN_DIM,
            num_layers=config.TRANSFORMER_LAYERS,
            num_heads=config.TRANSFORMER_HEADS,
            dropout=config.DROPOUT
        )
        
        # Parámetro de temperatura para la función de pérdida
        self.temperature = nn.Parameter(torch.tensor(config.TEMPERATURE))
        
    def encode_image(self, pixel_values):
        """
        Codifica una imagen usando CLIP y luego refina el embedding.
        
        Args:
            pixel_values: Valores de píxeles normalizados [batch_size, 3, H, W]
            
        Returns:
            Embedding refinado [batch_size, embed_dim]
        """
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            
        # Refinar con transformer
        refined_features = self.transformer_projection(image_features)
        
        return refined_features
    
    def encode_text(self, text):
        """
        Codifica texto usando CLIP.
        
        Args:
            text: Lista de strings con descripciones
            
        Returns:
            Embedding de texto [batch_size, embed_dim]
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip.get_text_features(**inputs)
            
        # Refinar con transformer
        refined_features = self.transformer_projection(text_features)
        
        return refined_features
    
    def forward(self, original_image, generated_image, negative_image, 
                original_text=None, generated_text=None, negative_text=None):
        """
        Forward pass del modelo siamés.
        
        Args:
            original_image: Imagen original [batch_size, 3, H, W]
            generated_image: Imagen generada [batch_size, 3, H, W]
            negative_image: Imagen negativa [batch_size, 3, H, W]
            original_text: (Opcional) Descripción de la imagen original
            generated_text: (Opcional) Descripción de la imagen generada
            negative_text: (Opcional) Descripción de la imagen negativa
            
        Returns:
            Embeddings refinados y similitudes
        """
        # Codificar imágenes
        original_embedding = self.encode_image(original_image)
        generated_embedding = self.encode_image(generated_image)
        negative_embedding = self.encode_image(negative_image)
        
        # Calcular similitudes del coseno
        sim_pos = F.cosine_similarity(original_embedding, generated_embedding, dim=1)
        sim_neg = F.cosine_similarity(original_embedding, negative_embedding, dim=1)
        
        # Si se proporcionan textos, también los codificamos
        text_embeddings = None
        if original_text is not None and Config.USE_TEXT_EMBEDDINGS:
            original_text_embedding = self.encode_text(original_text)
            generated_text_embedding = self.encode_text(generated_text)
            negative_text_embedding = self.encode_text(negative_text)
            
            text_embeddings = {
                'original': original_text_embedding,
                'generated': generated_text_embedding,
                'negative': negative_text_embedding
            }
        
        return {
            'embeddings': {
                'original': original_embedding,
                'generated': generated_embedding,
                'negative': negative_embedding,
                'text': text_embeddings
            },
            'similarities': {
                'positive': sim_pos,
                'negative': sim_neg
            }
        }