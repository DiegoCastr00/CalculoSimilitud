import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from transformers import CLIPFeatureExtractor

from config import Config
from model import SiameseCLIPModel

class ArtSimilarityInference:
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
        
        # Cargar checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Inicializar modelo
        self.model = SiameseCLIPModel(config)
        
        # Cargar pesos del modelo
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Inicializar extractor de características
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(config.CLIP_MODEL_NAME)
        
        # Crear directorio para resultados
        self.results_dir = Path("inference_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def preprocess_image(self, image_path):
        """
        Preprocesa una imagen para el modelo.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tensor de imagen procesado
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def extract_embedding(self, image_path):
        """
        Extrae el embedding de una imagen.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Embedding normalizado
        """
        pixel_values = self.preprocess_image(image_path)
        
        with torch.no_grad():
            embedding = self.model.encode_image(pixel_values)
        
        return embedding
    
    def compute_similarity(self, img1_path, img2_path, visualize=False):
        """
        Calcula la similitud entre dos imágenes.
        
        Args:
            img1_path: Ruta a la primera imagen
            img2_path: Ruta a la segunda imagen
            visualize: Si se debe visualizar la comparación
            
        Returns:
            Valor de similitud entre 0 y 1
        """
        # Extraer embeddings
        embedding1 = self.extract_embedding(img1_path)
        embedding2 = self.extract_embedding(img2_path)
        
        # Calcular similitud del coseno
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        
        # Normalizar a rango [0, 1]
        similarity = (similarity + 1) / 2
        
        if visualize:
            self.visualize_comparison(img1_path, img2_path, similarity)
        
        return similarity
    
    def visualize_comparison(self, img1_path, img2_path, similarity):
        """
        Visualiza la comparación entre dos imágenes.
        
        Args:
            img1_path: Ruta a la primera imagen
            img2_path: Ruta a la segunda imagen
            similarity: Valor de similitud calculado
        """
        # Cargar imágenes
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Mostrar imágenes
        ax1.imshow(img1)
        ax1.set_title("Imagen 1")
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.set_title("Imagen 2")
        ax2.axis('off')
        
        # Añadir similitud como título global
        fig.suptitle(f"Similitud: {similarity:.4f}", fontsize=16)
        
        # Ajustar espaciado
        plt.tight_layout()
        
        # Guardar figura
        output_path = self.results_dir / f"comparison_{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Comparación guardada en: {output_path}")
    
    def process_folder(self, reference_path, comparison_folder):
        """
        Compara una imagen de referencia con todas las imágenes en una carpeta.
        
        Args:
            reference_path: Ruta a la imagen de referencia
            comparison_folder: Ruta a la carpeta con imágenes para comparar
            
        Returns:
            Lista de tuplas (ruta_imagen, similitud) ordenada por similitud
        """
        # Extraer embedding de referencia
        reference_embedding = self.extract_embedding(reference_path)
        
        results = []
        
        # Obtener archivos de imagen en la carpeta
        image_extensions = ['.jpg', '.jpeg', '.png']
        comparison_paths = [
            os.path.join(comparison_folder, f) 
            for f in os.listdir(comparison_folder) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        # Calcular similitudes
        for img_path in comparison_paths:
            try:
                img_embedding = self.extract_embedding(img_path)
                similarity = F.cosine_similarity(reference_embedding, img_embedding).item()
                similarity = (similarity + 1) / 2  # Normalizar a [0, 1]
                results.append((img_path, similarity))
            except Exception as e:
                print(f"Error al procesar {img_path}: {e}")
        
        # Ordenar por similitud descendente
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def visualize_similar_images(self, reference_path, results, top_n=5):
        """
        Visualiza las imágenes más similares a una referencia.
        
        Args:
            reference_path: Ruta a la imagen de referencia
            results: Lista de tuplas (ruta_imagen, similitud)
            top_n: Número de imágenes similares a mostrar
        """
        # Cargar imagen de referencia
        ref_img = Image.open(reference_path).convert('RGB')
        
        # Limitar al top_n
        results = results[:top_n]
        
        # Crear figura
        fig, axes = plt.subplots(1, top_n + 1, figsize=(15, 5))
        
        # Mostrar imagen de referencia
        axes[0].imshow(ref_img)
        axes[0].set_title("Referencia")
        axes[0].axis('off')
        
        # Mostrar imágenes similares
        for i, (img_path, similarity) in enumerate(results):
            img = Image.open(img_path).convert('RGB')
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Sim: {similarity:.4f}")
            axes[i + 1].axis('off')
        
        # Ajustar espaciado
        plt.tight_layout()
        
        # Guardar figura
        output_path = self.results_dir / f"similar_to_{os.path.basename(reference_path).split('.')[0]}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Visualización guardada en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Inferencia con modelo de similitud artística")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al checkpoint del modelo")
    parser.add_argument("--mode", type=str, choices=['compare', 'search'], required=True, help="Modo de inferencia")
    parser.add_argument("--reference", type=str, required=True, help="Ruta a la imagen de referencia")
    parser.add_argument("--comparison", type=str, help="Ruta a la segunda imagen o carpeta para comparar")
    parser.add_argument("--top_n", type=int, default=5, help="Número de imágenes similares a mostrar")
    
    args = parser.parse_args()
    
    # Inicializar inferencia
    inference = ArtSimilarityInference(args.checkpoint, Config)
    
    if args.mode == 'compare':
        # Modo de comparación entre dos imágenes
        if not args.comparison:
            raise ValueError("El modo 'compare' requiere el argumento --comparison")
        
        print(f"Comparando {args.reference} con {args.comparison}...")
        similarity = inference.compute_similarity(args.reference, args.comparison, visualize=True)
        print(f"Similitud: {similarity:.4f}")
    
    elif args.mode == 'search':
        # Modo de búsqueda en carpeta
        if not args.comparison:
            raise ValueError("El modo 'search' requiere el argumento --comparison")
        if not os.path.isdir(args.comparison):
            raise ValueError(f"El directorio {args.comparison} no existe")
        
        print(f"Buscando imágenes similares a {args.reference} en {args.comparison}...")
        results = inference.process_folder(args.reference, args.comparison)
        
        # Imprimir resultados
        print(f"Top {min(args.top_n, len(results))} imágenes más similares:")
        for i, (img_path, similarity) in enumerate(results[:args.top_n]):
            print(f"{i+1}. {os.path.basename(img_path)}: {similarity:.4f}")
        
        # Visualizar resultados
        inference.visualize_similar_images(args.reference, results, top_n=args.top_n)

if __name__ == "__main__":
    main()