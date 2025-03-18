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
        self.register_buffer('min_temperature', torch.tensor(0.01))

        
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
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Codifica texto usando CLIP con input_ids ya procesados.
        
        Args:
            input_ids: Tokens de entrada [batch_size, seq_len]
            attention_mask: Máscara de atención [batch_size, seq_len]
            
        Returns:
            Embedding de texto refinado [batch_size, embed_dim]
        """
        with torch.no_grad():
            text_features = self.clip.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask if attention_mask is not None else None
            )
            
        # Refinar con transformer
        refined_features = self.transformer_projection(text_features)
        
        return refined_features
    
    def forward(self, batch):
        """
        Forward pass del modelo siamés.
        
        Args:
            batch: Diccionario con las claves:
                - original_pixel_values, generated_pixel_values, negative_pixel_values
                - original_input_ids, generated_input_ids, negative_input_ids (opcionales)
                - original_attention_mask, generated_attention_mask, negative_attention_mask (opcionales)
                
        Returns:
            Embeddings refinados y similitudes
        """
        # Codificar imágenes
        original_embedding = self.encode_image(batch['original_pixel_values'])
        generated_embedding = self.encode_image(batch['generated_pixel_values'])
        negative_embedding = self.encode_image(batch['negative_pixel_values'])
        
        # Si se proporcionan textos, también los codificamos
        text_embeddings = None
        if 'original_input_ids' in batch and Config.USE_TEXT_EMBEDDINGS:
            original_text_embedding = self.encode_text(
                batch['original_input_ids'], 
                batch.get('original_attention_mask')
            )
            generated_text_embedding = self.encode_text(
                batch['generated_input_ids'], 
                batch.get('generated_attention_mask')
            )
            negative_text_embedding = self.encode_text(
                batch['negative_input_ids'], 
                batch.get('negative_attention_mask')
            )
            
            # Combinar embeddings de imagen y texto (promedio simple)
            if not hasattr(self, 'fusion_layer'):
                self.fusion_layer = nn.Sequential(
                    nn.Linear(self.embed_dim * 2, self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(Config.DROPOUT),
                    nn.Linear(self.embed_dim, self.embed_dim),
                ).to(original_embedding.device)

            original_combined = torch.cat([original_embedding, original_text_embedding], dim=1)
            generated_combined = torch.cat([generated_embedding, generated_text_embedding], dim=1)
            negative_combined = torch.cat([negative_embedding, negative_text_embedding], dim=1)

            original_embedding = F.normalize(self.fusion_layer(original_combined), p=2, dim=1)
            generated_embedding = F.normalize(self.fusion_layer(generated_combined), p=2, dim=1)
            negative_embedding = F.normalize(self.fusion_layer(negative_combined), p=2, dim=1)
            
            text_embeddings = {
                'original': original_text_embedding,
                'generated': generated_text_embedding,
                'negative': negative_text_embedding
            }
        
        # Calcular similitudes del coseno
        sim_pos = F.cosine_similarity(original_embedding, generated_embedding, dim=1)
        sim_neg = F.cosine_similarity(original_embedding, negative_embedding, dim=1)
        
        temperature = torch.max(self.temperature, self.min_temperature)

        
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
            },
            'temperature': temperature

        }
    def calculate_similarity(self, image1_pixel_values, image2_pixel_values, 
                            text1_input_ids=None, text2_input_ids=None,
                            text1_attention_mask=None, text2_attention_mask=None):
        """
        Calcula similitud entre dos imágenes (y opcionalmente sus textos).
        
        Args:
            image1_pixel_values, image2_pixel_values: Tensores de imágenes procesadas
            text1_input_ids, text2_input_ids: IDs de tokens de texto (opcional)
            text1_attention_mask, text2_attention_mask: Máscaras de atención (opcional)
            
        Returns:
            Similitud del coseno entre los embeddings [-1, 1]
        """
        # Codificar imágenes
        embedding1 = self.encode_image(image1_pixel_values)
        embedding2 = self.encode_image(image2_pixel_values)
        
        # Si se proporcionan textos, también los codificamos y combinamos
        if text1_input_ids is not None and text2_input_ids is not None and Config.USE_TEXT_EMBEDDINGS:
            text_embedding1 = self.encode_text(text1_input_ids, text1_attention_mask)
            text_embedding2 = self.encode_text(text2_input_ids, text2_attention_mask)
            
            # Usar la capa de fusión para combinar si existe
            if hasattr(self, 'fusion_layer'):
                combined1 = torch.cat([embedding1, text_embedding1], dim=1)
                combined2 = torch.cat([embedding2, text_embedding2], dim=1)
                
                embedding1 = F.normalize(self.fusion_layer(combined1), p=2, dim=1)
                embedding2 = F.normalize(self.fusion_layer(combined2), p=2, dim=1)
            else:
                # Fallback al promedio simple
                embedding1 = F.normalize((embedding1 + text_embedding1) / 2, p=2, dim=1)
                embedding2 = F.normalize((embedding2 + text_embedding2) / 2, p=2, dim=1)
        
        # Calcular similitud del coseno
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        
        return similarity