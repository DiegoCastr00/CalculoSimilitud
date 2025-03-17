import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from config import Config
from model import SiameseCLIPModel
from dataset import ArtworkSimilarityDataset
from torch.utils.data import DataLoader

class ModelEvaluator:
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
        
        # Crear directorio para resultados
        self.results_dir = Path("results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def compute_similarity(self, img1_path, img2_path):
        """
        Calcula la similitud entre dos imágenes.
        
        Args:
            img1_path: Ruta a la primera imagen
            img2_path: Ruta a la segunda imagen
            
        Returns:
            Valor de similitud entre 0 y 1
        """
        from PIL import Image
        from transformers import CLIPFeatureExtractor
        
        # Preparar extractor de características
        feature_extractor = CLIPFeatureExtractor.from_pretrained(self.config.CLIP_MODEL_NAME)
        
        # Cargar imágenes
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Procesar imágenes
        inputs1 = feature_extractor(images=img1, return_tensors="pt")
        inputs2 = feature_extractor(images=img2, return_tensors="pt")
        
        # Mover a dispositivo
        pixel_values1 = inputs1['pixel_values'].to(self.device)
        pixel_values2 = inputs2['pixel_values'].to(self.device)
        
        # Calcular embeddings
        with torch.no_grad():
            embedding1 = self.model.encode_image(pixel_values1)
            embedding2 = self.model.encode_image(pixel_values2)
        
        # Calcular similitud del coseno
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        
        # Normalizar a rango [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def evaluate_dataset(self, dataloader):
        """
        Evalúa el modelo en un conjunto de datos y calcula métricas.
        
        Args:
            dataloader: DataLoader con los datos de evaluación
            
        Returns:
            Diccionario con métricas
        """
        similarities_pos = []
        similarities_neg = []
        triplet_correct = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Mover datos a GPU
                original_image = batch['original_image'].to(self.device)
                generated_image = batch['generated_image'].to(self.device)
                negative_image = batch['negative_image'].to(self.device)
                
                # Forward pass
                outputs = self.model(original_image, generated_image, negative_image)
                
                # Obtener similitudes
                sim_pos = outputs['similarities']['positive']
                sim_neg = outputs['similarities']['negative']
                
                # Normalizar a [0, 1]
                sim_pos = (sim_pos + 1) / 2
                sim_neg = (sim_neg + 1) / 2
                
                # Guardar similitudes
                similarities_pos.extend(sim_pos.cpu().numpy())
                similarities_neg.extend(sim_neg.cpu().numpy())
                
                # Contar tripletes correctos
                triplet_correct += (sim_pos > sim_neg).sum().item()
        
        # Calcular métricas
        triplet_accuracy = triplet_correct / len(dataloader.dataset)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'similar_pair': similarities_pos,
            'dissimilar_pair': similarities_neg
        })
        
        # Guardar resultados
        results.to_csv(self.results_dir / "similarity_results.csv", index=False)
        
        return {
            'triplet_accuracy': triplet_accuracy,
            'mean_similar': np.mean(similarities_pos),
            'mean_dissimilar': np.mean(similarities_neg),
            'std_similar': np.std(similarities_pos),
            'std_dissimilar': np.std(similarities_neg),
            'results_df': results
        }
    
    def plot_similarity_distributions(self, metrics):
        """
        Genera gráficos de distribución de similitudes.
        
        Args:
            metrics: Diccionario con métricas y DataFrame de resultados
        """
        results_df = metrics['results_df']
        
        # Configurar estilo
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Trazar distribuciones
        sns.histplot(data=results_df, x="similar_pair", color="blue", alpha=0.5, label="Original-Generado")
        sns.histplot(data=results_df, x="dissimilar_pair", color="red", alpha=0.5, label="Original-Negativo")
        
        # Añadir líneas verticales para medias
        plt.axvline(metrics['mean_similar'], color='blue', linestyle='--')
        plt.axvline(metrics['mean_dissimilar'], color='red', linestyle='--')
        
        # Configurar gráfico
        plt.title("Distribución de Similitudes entre Pares de Imágenes")
        plt.xlabel("Similitud del Coseno (Normalizada)")
        plt.ylabel("Frecuencia")
        plt.legend()
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.results_dir / "similarity_distributions.png", dpi=300)
        plt.close()
    
    def generate_similarity_matrix(self, images_paths, labels=None):
        """
        Genera una matriz de similitud entre un conjunto de imágenes.
        
        Args:
            images_paths: Lista de rutas a imágenes
            labels: Lista de etiquetas para las imágenes
        """
        n = len(images_paths)
        similarity_matrix = np.zeros((n, n))
        
        # Calcular similitudes
        for i in tqdm(range(n), desc="Generating similarity matrix"):
            for j in range(i, n):
                sim = self.compute_similarity(images_paths[i], images_paths[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Crear mapa de calor
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        
        # Usar etiquetas si se proporcionan
        if labels is None:
            labels = [os.path.basename(path) for path in images_paths]
        
        # Generar mapa de calor
        ax = sns.heatmap(
            similarity_matrix,
            cmap="viridis",
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            square=True,
            mask=mask,
            xticklabels=labels,
            yticklabels=labels
        )
        
        # Rotar etiquetas
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.title("Matriz de Similitud entre Imágenes")
        plt.tight_layout()
        plt.savefig(self.results_dir / "similarity_matrix.png", dpi=300)
        plt.close()
        
        return similarity_matrix

def main():
    parser = argparse.ArgumentParser(description="Evaluación del modelo de similitud artística")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al checkpoint del modelo")
    parser.add_argument("--data_csv", type=str, default=None, help="Ruta al CSV de evaluación")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--sample_images", type=str, default=None, help="Ruta al CSV con imágenes para matriz de similitud")
    
    args = parser.parse_args()
    
    # Actualizar configuración
    if args.data_csv:
        Config.DATA_CSV = args.data_csv
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    # Inicializar evaluador
    evaluator = ModelEvaluator(args.checkpoint, Config)
    
    # Evaluar en dataset
    dataset = ArtworkSimilarityDataset(Config.DATA_CSV, mode='val')
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    print("Evaluando modelo en el conjunto de datos...")
    metrics = evaluator.evaluate_dataset(dataloader)
    
    # Imprimir métricas
    print(f"Triplet Accuracy: {metrics['triplet_accuracy']:.4f}")
    print(f"Mean Similar Pair Similarity: {metrics['mean_similar']:.4f} ± {metrics['std_similar']:.4f}")
    print(f"Mean Dissimilar Pair Similarity: {metrics['mean_dissimilar']:.4f} ± {metrics['std_dissimilar']:.4f}")
    
    # Generar gráficos de distribución
    print("Generando gráficos de distribución...")
    evaluator.plot_similarity_distributions(metrics)
    
    # Generar matriz de similitud si se proporciona una lista de imágenes
    if args.sample_images:
        print("Generando matriz de similitud...")
        sample_df = pd.read_csv(args.sample_images)
        image_paths = [os.path.join(Config.DATA_ROOT, path) for path in sample_df['image_path']]
        labels = sample_df['label'] if 'label' in sample_df.columns else None
        
        evaluator.generate_similarity_matrix(image_paths, labels)

if __name__ == "__main__":
    main()