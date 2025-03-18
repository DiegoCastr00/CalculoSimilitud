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
    
    def calculate_distribution_metrics(self, metrics):
        """
        Calcula métricas de separación entre distribuciones de similitud.
        """
        similar = np.array(metrics['results_df']['similar_pair'])
        dissimilar = np.array(metrics['results_df']['dissimilar_pair'])
        
        # Distancia entre medias
        mean_separation = metrics['mean_similar'] - metrics['mean_dissimilar']
        
        # Coeficiente de separabilidad (d-prime)
        pooled_std = np.sqrt((metrics['std_similar']**2 + metrics['std_dissimilar']**2) / 2)
        d_prime = mean_separation / pooled_std if pooled_std > 0 else float('inf')
        
        # Área bajo la curva (AUC) aproximada
        from sklearn.metrics import roc_auc_score
        y_true = np.concatenate([np.ones(len(similar)), np.zeros(len(dissimilar))])
        y_score = np.concatenate([similar, dissimilar])
        auc = roc_auc_score(y_true, y_score)
        
        return {
            'mean_separation': mean_separation,
            'd_prime': d_prime,
            'auc': auc
        }
    def find_optimal_threshold(self, metrics):
        """
        Encuentra el umbral óptimo para clasificar pares similares vs. disimilares.
        """
        from sklearn.metrics import precision_recall_curve, f1_score
        
        similar = np.array(metrics['results_df']['similar_pair'])
        dissimilar = np.array(metrics['results_df']['dissimilar_pair'])
        
        # Crear valores de verdad y puntuaciones
        y_true = np.concatenate([np.ones(len(similar)), np.zeros(len(dissimilar))])
        y_score = np.concatenate([similar, dissimilar])
        
        # Calcular precision-recall para diferentes umbrales
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        # Calcular F1 score para cada umbral
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Encontrar umbral óptimo
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Calcular métricas con umbral óptimo
        y_pred = (y_score >= optimal_threshold).astype(int)
        optimal_f1 = f1_score(y_true, y_pred)
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1
        }
    def plot_roc_and_pr_curves(self, metrics):
        """
        Genera curvas ROC y Precision-Recall.
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        similar = np.array(metrics['results_df']['similar_pair'])
        dissimilar = np.array(metrics['results_df']['dissimilar_pair'])
        
        # Preparar datos
        y_true = np.concatenate([np.ones(len(similar)), np.zeros(len(dissimilar))])
        y_score = np.concatenate([similar, dissimilar])
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Calcular curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        # Graficar curva ROC
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Graficar curva Precision-Recall
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "roc_pr_curves.png", dpi=300)
        plt.close()
        
        return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

    def analyze_edge_cases(self, dataloader, n_examples=5):
        """
        Analiza los ejemplos en la frontera (más difíciles de clasificar).
        """
        from torchvision.utils import save_image
        edge_cases_dir = self.results_dir / "edge_cases"
        os.makedirs(edge_cases_dir, exist_ok=True)
        
        all_results = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Finding edge cases")):
                # Procesar batch
                original_image = batch['original_image'].to(self.device)
                generated_image = batch['generated_image'].to(self.device)
                negative_image = batch['negative_image'].to(self.device)
                
                # Obtener paths de imágenes
                original_paths = batch['original_path']
                generated_paths = batch['generated_path']
                negative_paths = batch['negative_path']
                
                # Forward pass
                outputs = self.model(original_image, generated_image, negative_image)
                
                # Obtener similitudes
                sim_pos = outputs['similarities']['positive']
                sim_neg = outputs['similarities']['negative']
                
                # Normalizar a [0, 1]
                sim_pos = (sim_pos + 1) / 2
                sim_neg = (sim_neg + 1) / 2
                
                # Calcular diferencia
                diff = sim_pos - sim_neg
                
                # Guardar resultados
                for j in range(len(original_image)):
                    all_results.append({
                        'original_path': original_paths[j],
                        'generated_path': generated_paths[j],
                        'negative_path': negative_paths[j],
                        'sim_pos': sim_pos[j].item(),
                        'sim_neg': sim_neg[j].item(),
                        'diff': diff[j].item()
                    })
        
        # Ordenar por diferencia (casos más difíciles primero)
        all_results.sort(key=lambda x: abs(x['diff']))
        
        # Guardar ejemplos más difíciles
        for i, case in enumerate(all_results[:n_examples]):
            # Guardar información
            with open(edge_cases_dir / f"case_{i+1}_info.txt", 'w') as f:
                f.write(f"Original: {case['original_path']}\n")
                f.write(f"Generated: {case['generated_path']}\n")
                f.write(f"Negative: {case['negative_path']}\n")
                f.write(f"Similarity (original-generated): {case['sim_pos']:.4f}\n")
                f.write(f"Similarity (original-negative): {case['sim_neg']:.4f}\n")
                f.write(f"Difference: {case['diff']:.4f}\n")
            
            # Guardar imágenes
            from PIL import Image
            original = Image.open(case['original_path']).convert('RGB')
            generated = Image.open(case['generated_path']).convert('RGB')
            negative = Image.open(case['negative_path']).convert('RGB')
            
            original.save(edge_cases_dir / f"case_{i+1}_original.jpg")
            generated.save(edge_cases_dir / f"case_{i+1}_generated.jpg")
            negative.save(edge_cases_dir / f"case_{i+1}_negative.jpg")
        
        return all_results[:n_examples]

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
    parser.add_argument("--analyze_edge_cases", action="store_true", help="Analizar casos difíciles")
    parser.add_argument("--eval_consistency", action="store_true", help="Evaluar consistencia de métricas")
    
    args = parser.parse_args()
    
    # Actualizar configuración...
    
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
    
    # Imprimir métricas básicas
    print(f"Triplet Accuracy: {metrics['triplet_accuracy']:.4f}")
    print(f"Mean Similar Pair Similarity: {metrics['mean_similar']:.4f} ± {metrics['std_similar']:.4f}")
    print(f"Mean Dissimilar Pair Similarity: {metrics['mean_dissimilar']:.4f} ± {metrics['std_dissimilar']:.4f}")
    
    # Calcular métricas de distribución
    print("Calculando métricas de separación de distribuciones...")
    dist_metrics = evaluator.calculate_distribution_metrics(metrics)
    print(f"Mean Separation: {dist_metrics['mean_separation']:.4f}")
    print(f"D-prime: {dist_metrics['d_prime']:.4f}")
    print(f"AUC: {dist_metrics['auc']:.4f}")
    
    # Encontrar umbral óptimo
    print("Calculando umbral óptimo...")
    threshold_metrics = evaluator.find_optimal_threshold(metrics)
    print(f"Optimal Threshold: {threshold_metrics['optimal_threshold']:.4f}")
    print(f"Optimal F1 Score: {threshold_metrics['optimal_f1']:.4f}")
    
    # Generar curvas ROC y PR
    print("Generando curvas ROC y Precision-Recall...")
    curve_metrics = evaluator.plot_roc_and_pr_curves(metrics)
    print(f"ROC AUC: {curve_metrics['roc_auc']:.4f}")
    print(f"PR AUC: {curve_metrics['pr_auc']:.4f}")
    
    # Generar gráficos de distribución
    print("Generando gráficos de distribución...")
    evaluator.plot_similarity_distributions(metrics)
    
    # Analizar casos difíciles
    if args.analyze_edge_cases:
        print("Analizando casos difíciles...")
        edge_cases = evaluator.analyze_edge_cases(dataloader)
    
    # Evaluar consistencia
    if args.eval_consistency:
        print("Evaluando consistencia frente a perturbaciones...")
        consistency_metrics = evaluator.evaluate_metric_consistency(dataloader)
        print("Coeficientes de variación:")
        for perturb, cv in consistency_metrics['coefficient_of_variation'].items():
            print(f"  {perturb}: {cv:.4f}")
    
    # Generar matriz de similitud si se proporciona una lista de imágenes
    if args.sample_images:
        print("Generando matriz de similitud...")
        sample_df = pd.read_csv(args.sample_images)
        image_paths = [os.path.join(Config.DATA_ROOT, path) for path in sample_df['image_path']]
        labels = sample_df['label'] if 'label' in sample_df.columns else None
        
        evaluator.generate_similarity_matrix(image_paths, labels)
    
    # Guardar todas las métricas
    print("Guardando métricas completas...")
    all_metrics = {
        **metrics,
        'distribution_metrics': dist_metrics,
        'threshold_metrics': threshold_metrics,
        'curve_metrics': curve_metrics
    }
    
    # Eliminar DataFrame para poder guardar en JSON
    if 'results_df' in all_metrics:
        del all_metrics['results_df']
    
    import json
    with open(evaluator.results_dir / "all_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()