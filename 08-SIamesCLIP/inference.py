import torch
import torch.nn.functional as F
from PIL import Image
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor

from model import SiameseCLIPModel
from config import Config

class SimilarityInference:
    def __init__(self, checkpoint_path, device=None):
        """
        Inicializa el modelo para inferencia.
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo entrenado
            device: Dispositivo para inferencia (cuda o cpu)
        """
        # Configurar dispositivo
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Utilizando dispositivo: {self.device}")
        
        # Cargar configuración
        self.config = Config
        
        # Inicializar modelo
        self.model = SiameseCLIPModel(self.config)
        
        # Cargar checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Mover modelo al dispositivo
        self.model.to(self.device)
        self.model.eval()
        
        # Inicializar procesador de CLIP
        self.processor = CLIPProcessor.from_pretrained(self.config.CLIP_MODEL_NAME)
        
    def _load_checkpoint(self, checkpoint_path):
        """
        Carga los pesos del modelo desde un checkpoint.
        
        Args:
            checkpoint_path: Ruta al checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Cargar estado del modelo
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo cargado desde {checkpoint_path}")
            print(f"Época: {checkpoint.get('epoch', 'N/A')}")
            print(f"Mejor pérdida de validación: {checkpoint.get('best_val_loss', 'N/A')}")
        else:
            # Si el checkpoint solo contiene los pesos del modelo
            self.model.load_state_dict(checkpoint)
            print(f"Modelo cargado desde {checkpoint_path} (formato simple)")
    
    def preprocess_image(self, image_path):
        """
        Preprocesa una imagen para inferencia.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tensor de la imagen procesada
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
        
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        
        # Procesar imagen con el procesador de CLIP
        inputs = self.processor(images=image, return_tensors="pt")
        
        return inputs.pixel_values.to(self.device)
    
    def preprocess_text(self, text):
        """
        Preprocesa un texto para inferencia.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Tensores de input_ids y attention_mask
        """
        # Procesar texto con el procesador de CLIP
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        
        return {
            'input_ids': inputs.input_ids.to(self.device),
            'attention_mask': inputs.attention_mask.to(self.device)
        }
    
    def calculate_similarity(self, image1_path, image2_path, text1=None, text2=None):
        """
        Calcula la similitud entre dos imágenes (y opcionalmente sus textos).
        
        Args:
            image1_path: Ruta a la primera imagen
            image2_path: Ruta a la segunda imagen
            text1: Descripción de la primera imagen (opcional)
            text2: Descripción de la segunda imagen (opcional)
            
        Returns:
            Similitud del coseno entre los embeddings [-1, 1]
        """
        # Preprocesar imágenes
        image1 = self.preprocess_image(image1_path)
        image2 = self.preprocess_image(image2_path)
        
        # Preprocesar textos si se proporcionan
        text1_inputs = None
        text2_inputs = None
        
        if text1 is not None and text2 is not None and self.config.USE_TEXT_EMBEDDINGS:
            text1_inputs = self.preprocess_text(text1)
            text2_inputs = self.preprocess_text(text2)
        
        # Calcular similitud
        with torch.no_grad():
            if text1_inputs is not None and text2_inputs is not None:
                similarity = self.model.calculate_similarity(
                    image1_pixel_values=image1,
                    image2_pixel_values=image2,
                    text1_input_ids=text1_inputs['input_ids'],
                    text2_input_ids=text2_inputs['input_ids'],
                    text1_attention_mask=text1_inputs['attention_mask'],
                    text2_attention_mask=text2_inputs['attention_mask']
                )
            else:
                similarity = self.model.calculate_similarity(
                    image1_pixel_values=image1,
                    image2_pixel_values=image2
                )
        
        return similarity.item()
    
    def calculate_batch_similarities(self, reference_image, comparison_images, reference_text=None, comparison_texts=None):
        """
        Calcula similitudes entre una imagen de referencia y múltiples imágenes de comparación.
        
        Args:
            reference_image: Ruta a la imagen de referencia
            comparison_images: Lista de rutas a imágenes para comparar
            reference_text: Descripción de la imagen de referencia (opcional)
            comparison_texts: Lista de descripciones para las imágenes de comparación (opcional)
            
        Returns:
            Lista de similitudes ordenadas de mayor a menor
        """
        # Preprocesar imagen de referencia
        ref_image = self.preprocess_image(reference_image)
        
        # Preprocesar texto de referencia si se proporciona
        ref_text_inputs = None
        if reference_text is not None and self.config.USE_TEXT_EMBEDDINGS:
            ref_text_inputs = self.preprocess_text(reference_text)
        
        results = []
        
        # Calcular similitud para cada imagen de comparación
        for i, comp_image_path in enumerate(comparison_images):
            try:
                # Preprocesar imagen de comparación
                comp_image = self.preprocess_image(comp_image_path)
                
                # Preprocesar texto de comparación si se proporciona
                comp_text_inputs = None
                if comparison_texts is not None and i < len(comparison_texts) and self.config.USE_TEXT_EMBEDDINGS:
                    comp_text_inputs = self.preprocess_text(comparison_texts[i])
                
                # Calcular similitud
                with torch.no_grad():
                    if ref_text_inputs is not None and comp_text_inputs is not None:
                        similarity = self.model.calculate_similarity(
                            image1_pixel_values=ref_image,
                            image2_pixel_values=comp_image,
                            text1_input_ids=ref_text_inputs['input_ids'],
                            text2_input_ids=comp_text_inputs['input_ids'],
                            text1_attention_mask=ref_text_inputs['attention_mask'],
                            text2_attention_mask=comp_text_inputs['attention_mask']
                        )
                    else:
                        similarity = self.model.calculate_similarity(
                            image1_pixel_values=ref_image,
                            image2_pixel_values=comp_image
                        )
                
                results.append({
                    'image_path': comp_image_path,
                    'similarity': similarity.item()
                })
            except Exception as e:
                print(f"Error al procesar {comp_image_path}: {e}")
        
        # Ordenar resultados por similitud (de mayor a menor)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def visualize_similarities(self, reference_image, comparison_results, num_images=5, figsize=(15, 10)):
        """
        Visualiza las similitudes entre una imagen de referencia y las imágenes más similares.
        
        Args:
            reference_image: Ruta a la imagen de referencia
            comparison_results: Resultados de calculate_batch_similarities
            num_images: Número de imágenes similares a mostrar
            figsize: Tamaño de la figura
        """
        # Limitar el número de imágenes a mostrar
        num_images = min(num_images, len(comparison_results))
        
        # Crear figura
        fig, axes = plt.subplots(1, num_images + 1, figsize=figsize)
        
        # Mostrar imagen de referencia
        ref_img = Image.open(reference_image).convert('RGB')
        axes[0].imshow(ref_img)
        axes[0].set_title("Imagen de referencia")
        axes[0].axis('off')
        
        # Mostrar imágenes similares
        for i in range(num_images):
            img_path = comparison_results[i]['image_path']
            similarity = comparison_results[i]['similarity']
            
            img = Image.open(img_path).convert('RGB')
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"Sim: {similarity:.4f}")
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()

