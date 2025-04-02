import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import torch.nn.functional as F
from collections import defaultdict

from inference import SimilarityInference

class SiameseModelEvaluator:
    """
    Clase para evaluar el rendimiento de un modelo siamés entrenado sin supervisión.
    Proporciona métricas y visualizaciones para analizar la calidad de los embeddings
    y la consistencia de las similitudes calculadas.
    """
    
    def __init__(self, model_inference, output_dir=None):
        """
        Inicializa el evaluador con un modelo de inferencia.
        
        Args:
            model_inference: Instancia de SimilarityInference inicializada con el modelo a evaluar
            output_dir: Directorio para guardar resultados (opcional)
        """
        self.model_inference = model_inference
        self.model = model_inference.model
        self.device = model_inference.device
        
        # Directorio para guardar resultados
        if output_dir is None:
            self.output_dir = Path("evaluation_results")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Almacenamiento para embeddings calculados
        self.embeddings_cache = {}
        self.image_paths = []
        self.embeddings = []
        
    def extract_embeddings(self, image_paths, batch_size=32, use_text=False, texts=None):
        """
        Extrae embeddings para un conjunto de imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            batch_size: Tamaño del lote para procesamiento
            use_text: Si se deben usar descripciones de texto junto con las imágenes
            texts: Lista de descripciones de texto (opcional, solo si use_text=True)
            
        Returns:
            Tensor con embeddings normalizados [n_images, embed_dim]
        """
        self.model.eval()
        all_embeddings = []
        self.image_paths = image_paths
        
        # Verificar si se proporcionan textos cuando use_text=True
        if use_text and (texts is None or len(texts) != len(image_paths)):
            raise ValueError("Si use_text=True, se debe proporcionar una lista de textos del mismo tamaño que image_paths")
        
        # Procesar en lotes
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extrayendo embeddings"):
            batch_paths = image_paths[i:i+batch_size]
            batch_embeddings = []
            
            for j, img_path in enumerate(batch_paths):
                # Preprocesar imagen
                img_tensor = self.model_inference.preprocess_image(img_path)
                
                with torch.no_grad():
                    # Si se usan textos, combinar con embeddings de imagen
                    if use_text and texts is not None:
                        text_inputs = self.model_inference.preprocess_text(texts[i+j])
                        embedding = self.model.encode_multimodal(
                            img_tensor, 
                            text_inputs['input_ids'],
                            text_inputs['attention_mask']
                        )
                    else:
                        # Solo usar embeddings de imagen
                        embedding = self.model.encode_image(img_tensor)
                
                batch_embeddings.append(embedding)
            
            # Concatenar embeddings del lote
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
            all_embeddings.append(batch_embeddings.cpu())
        
        # Concatenar todos los embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Normalizar para similitud del coseno
        normalized_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        
        self.embeddings = normalized_embeddings
        return normalized_embeddings
    
    def compute_similarity_matrix(self, embeddings=None):
        """
        Calcula la matriz de similitud entre todos los embeddings.
        
        Args:
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            
        Returns:
            Matriz de similitud [n_samples, n_samples]
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        # Calcular matriz de similitud del coseno
        similarity_matrix = torch.mm(embeddings, embeddings.t()).cpu().numpy()
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, similarity_matrix=None, save_path=None):
        """
        Visualiza la matriz de similitud como un mapa de calor.
        
        Args:
            similarity_matrix: Matriz de similitud [n_samples, n_samples] (opcional)
            save_path: Ruta para guardar la visualización (opcional)
        """
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix()
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)
        plt.title('Matriz de Similitud del Coseno')
        plt.xlabel('Índice de Imagen')
        plt.ylabel('Índice de Imagen')
        
        if save_path is None:
            save_path = self.output_dir / "similarity_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_embeddings_tsne(self, embeddings=None, perplexity=30, n_iter=1000, save_path=None):
        """
        Visualiza los embeddings usando t-SNE para reducción de dimensionalidad.
        
        Args:
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            perplexity: Parámetro de perplexidad para t-SNE
            n_iter: Número de iteraciones para t-SNE
            save_path: Ruta para guardar la visualización (opcional)
            
        Returns:
            Coordenadas t-SNE [n_samples, 2]
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        # Convertir a numpy para t-SNE
        embeddings_np = embeddings.cpu().numpy()
        
        # Aplicar t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(embeddings_np)
        
        # Visualizar resultados
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
        plt.title('Visualización t-SNE de Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if save_path is None:
            save_path = self.output_dir / "tsne_visualization.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return tsne_results
    
    def evaluate_clustering(self, embeddings=None, n_clusters_range=range(2, 11)):
        """
        Evalúa la calidad del clustering usando métricas internas.
        
        Args:
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            n_clusters_range: Rango de número de clusters a evaluar
            
        Returns:
            DataFrame con métricas de evaluación para diferentes números de clusters
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        # Convertir a numpy para clustering
        embeddings_np = embeddings.cpu().numpy()
        
        results = []
        
        for n_clusters in tqdm(n_clusters_range, desc="Evaluando clustering"):
            # Aplicar K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_np)
            
            # Calcular métricas de evaluación interna
            if len(np.unique(cluster_labels)) > 1:  # Asegurar que hay al menos 2 clusters
                silhouette = silhouette_score(embeddings_np, cluster_labels)
                davies_bouldin = davies_bouldin_score(embeddings_np, cluster_labels)
                
                # Calcular inercia (suma de distancias al cuadrado dentro del cluster)
                inertia = kmeans.inertia_
                
                results.append({
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'inertia': inertia
                })
        
        # Convertir a DataFrame
        results_df = pd.DataFrame(results)
        
        # Visualizar resultados
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        
        # Silhouette score (mayor es mejor)
        ax[0].plot(results_df['n_clusters'], results_df['silhouette_score'], 'o-')
        ax[0].set_title('Silhouette Score vs. Número de Clusters')
        ax[0].set_xlabel('Número de Clusters')
        ax[0].set_ylabel('Silhouette Score')
        ax[0].grid(True)
        
        # Davies-Bouldin score (menor es mejor)
        ax[1].plot(results_df['n_clusters'], results_df['davies_bouldin_score'], 'o-')
        ax[1].set_title('Davies-Bouldin Score vs. Número de Clusters')
        ax[1].set_xlabel('Número de Clusters')
        ax[1].set_ylabel('Davies-Bouldin Score')
        ax[1].grid(True)
        
        # Método del codo para K-means
        ax[2].plot(results_df['n_clusters'], results_df['inertia'], 'o-')
        ax[2].set_title('Método del Codo para K-means')
        ax[2].set_xlabel('Número de Clusters')
        ax[2].set_ylabel('Inercia')
        ax[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "clustering_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return results_df
    
    def visualize_clusters(self, embeddings=None, n_clusters=5, tsne_results=None, save_path=None):
        """
        Visualiza los clusters usando t-SNE y K-means.
        
        Args:
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            n_clusters: Número de clusters para K-means
            tsne_results: Resultados de t-SNE precalculados (opcional)
            save_path: Ruta para guardar la visualización (opcional)
            
        Returns:
            Etiquetas de cluster [n_samples]
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        # Convertir a numpy para clustering
        embeddings_np = embeddings.cpu().numpy()
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # Calcular t-SNE si no se proporciona
        if tsne_results is None:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            tsne_results = tsne.fit_transform(embeddings_np)
        
        # Visualizar clusters
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, 
                   cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Visualización de Clusters (K-means, n_clusters={n_clusters})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if save_path is None:
            save_path = self.output_dir / f"clusters_visualization_k{n_clusters}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cluster_labels
    
    def evaluate_retrieval(self, query_indices, top_k=5, embeddings=None, similarity_matrix=None):
        """
        Evalúa la calidad de la recuperación de imágenes similares.
        
        Args:
            query_indices: Índices de las imágenes de consulta
            top_k: Número de imágenes similares a recuperar
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            similarity_matrix: Matriz de similitud precalculada (opcional)
            
        Returns:
            Diccionario con resultados de recuperación para cada consulta
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix(embeddings)
            
        results = {}
        
        for idx in query_indices:
            # Obtener similitudes para la imagen de consulta
            similarities = similarity_matrix[idx]
            
            # Ordenar por similitud (excluyendo la propia imagen)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_indices = sorted_indices[sorted_indices != idx][:top_k]
            
            # Guardar resultados
            results[idx] = {
                'query_index': idx,
                'query_path': self.image_paths[idx] if idx < len(self.image_paths) else None,
                'similar_indices': sorted_indices.tolist(),
                'similar_paths': [self.image_paths[i] if i < len(self.image_paths) else None for i in sorted_indices],
                'similarity_scores': similarities[sorted_indices].tolist()
            }
            
        return results
    
    def visualize_retrieval_results(self, retrieval_results, save_dir=None):
        """
        Visualiza los resultados de recuperación de imágenes similares.
        
        Args:
            retrieval_results: Resultados de la función evaluate_retrieval
            save_dir: Directorio para guardar las visualizaciones (opcional)
            
        Returns:
            Lista de rutas a las visualizaciones guardadas
        """
        if save_dir is None:
            save_dir = self.output_dir / "retrieval_results"
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            
        saved_paths = []
        
        for query_idx, result in retrieval_results.items():
            query_path = result['query_path']
            similar_paths = result['similar_paths']
            similarity_scores = result['similarity_scores']
            
            # Verificar que las rutas existen
            if query_path is None or any(p is None for p in similar_paths):
                continue
                
            # Cargar imágenes
            try:
                query_img = Image.open(query_path).convert('RGB')
                similar_imgs = [Image.open(p).convert('RGB') for p in similar_paths]
            except Exception as e:
                print(f"Error al cargar imágenes para consulta {query_idx}: {e}")
                continue
                
            # Crear figura
            fig, axes = plt.subplots(1, len(similar_imgs) + 1, figsize=(15, 4))
            
            # Mostrar imagen de consulta
            axes[0].imshow(query_img)
            axes[0].set_title(f"Consulta\n{Path(query_path).name}")
            axes[0].axis('off')
            
            # Mostrar imágenes similares
            for i, (img, score) in enumerate(zip(similar_imgs, similarity_scores)):
                axes[i+1].imshow(img)
                axes[i+1].set_title(f"Sim: {score:.3f}\n{Path(similar_paths[i]).name}")
                axes[i+1].axis('off')
                
            plt.tight_layout()
            
            # Guardar figura
            save_path = save_dir / f"retrieval_query_{query_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(save_path)
            
        return saved_paths
    
    def compute_neighborhood_consistency(self, embeddings=None, k=5):
        """
        Calcula la consistencia del vecindario para evaluar la estabilidad de los embeddings.
        
        Args:
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            k: Número de vecinos a considerar
            
        Returns:
            Puntuación de consistencia del vecindario [0, 1]
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        # Convertir a numpy para KNN
        embeddings_np = embeddings.cpu().numpy()
        
        # Calcular vecinos más cercanos
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 porque el primer vecino es la propia muestra
        nn.fit(embeddings_np)
        distances, indices = nn.kneighbors(embeddings_np)
        
        # Eliminar la propia muestra
        indices = indices[:, 1:]
        
        # Calcular consistencia del vecindario
        consistency_scores = []
        
        for i in range(len(embeddings_np)):
            neighbors = indices[i]
            neighbor_consistency = []
            
            for neighbor in neighbors:
                # Verificar si el vecino también tiene a la muestra original como vecino
                if i in indices[neighbor]:
                    neighbor_consistency.append(1)
                else:
                    neighbor_consistency.append(0)
                    
            consistency_scores.append(np.mean(neighbor_consistency))
            
        # Puntuación global
        global_consistency = np.mean(consistency_scores)
        
        return {
            'global_consistency': global_consistency,
            'per_sample_consistency': consistency_scores
        }
    
    def evaluate_embedding_quality(self, embeddings=None):
        """
        Evalúa la calidad general de los embeddings usando varias métricas.
        
        Args:
            embeddings: Tensor de embeddings [n_samples, embed_dim] (opcional)
            
        Returns:
            Diccionario con métricas de calidad
        """
        if embeddings is None:
            if len(self.embeddings) == 0:
                raise ValueError("No hay embeddings disponibles. Ejecute extract_embeddings primero.")
            embeddings = self.embeddings
            
        # Convertir a numpy
        embeddings_np = embeddings.cpu().numpy()
        
        # 1. Calcular norma media (debería ser cercana a 1 si están normalizados)
        norms = np.linalg.norm(embeddings_np, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # 2. Calcular distribución de similitudes
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        np.fill_diagonal(similarity_matrix, 0)  # Ignorar similitud consigo mismo
        mean_similarity = np.mean(similarity_matrix)
        std_similarity = np.std(similarity_matrix)
        
        # 3. Calcular consistencia del vecindario
        neighborhood_consistency = self.compute_neighborhood_consistency(embeddings)
        
        # 4. Evaluar distribución de los embeddings
        # Calcular la distancia media al centroide
        centroid = np.mean(embeddings_np, axis=0)
        distances_to_centroid = np.linalg.norm(embeddings_np - centroid, axis=1)
        mean_distance = np.mean(distances_to_centroid)
        std_distance = np.std(distances_to_centroid)
        
        # Visualizar distribución de similitudes
        plt.figure(figsize=(10, 6))
        plt.hist(similarity_matrix.flatten(), bins=50, alpha=0.7)
        plt.title('Distribución de Similitudes del Coseno')
        plt.xlabel('Similitud del Coseno')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "similarity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualizar distribución de distancias al centroide
        plt.figure(figsize=(10, 6))
        plt.hist(distances_to_centroid, bins=50, alpha=0.7)
        plt.title('Distribución de Distancias al Centroide')
        plt.xlabel('Distancia Euclidiana')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "centroid_distances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'embedding_stats': {
                'mean_norm': mean_norm,
                'std_norm': std_norm,
                'expected_norm': 1.0  # Para embeddings normalizados
            },
            'similarity_stats': {
                'mean_similarity': mean_similarity,
                'std_similarity': std_similarity
            },
            'neighborhood_consistency': neighborhood_consistency['global_consistency'],
            'centroid_stats': {
                'mean_distance': mean_distance,
                'std_distance': std_distance
            }
        }
    
    def run_complete_evaluation(self, image_paths, batch_size=32, use_text=False, texts=None, 
                               n_clusters=5, top_k=5, query_indices=None, save_results=True):
        """
        Ejecuta una evaluación completa del modelo.
        
        Args:
            image_paths: Lista de rutas a imágenes
            batch_size: Tamaño del lote para procesamiento
            use_text: Si se deben usar descripciones de texto junto con las imágenes
            texts: Lista de descripciones de texto (opcional, solo si use_text=True)
            n_clusters: Número de clusters para K-means
            top_k: Número de imágenes similares a recuperar
            query_indices: Índices de las imágenes de consulta (opcional)
            save_results: Si se deben guardar los resultados
            
        Returns:
            Diccionario con todos los resultados de evaluación
        """
        print("Iniciando evaluación completa del modelo...")
        
        # 1. Extraer embeddings
        print("Extrayendo embeddings...")
        embeddings = self.extract_embeddings(image_paths, batch_size, use_text, texts)
        
        # 2. Calcular matriz de similitud
        print("Calculando matriz de similitud...")
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # 3. Visualizar matriz de similitud
        print("Visualizando matriz de similitud...")
        sim_matrix_path = self.visualize_similarity_matrix(similarity_matrix)
        
        # 4. Visualizar embeddings con t-SNE
        print("Visualizando embeddings con t-SNE...")
        tsne_results = self.visualize_embeddings_tsne(embeddings)
        
        # 5. Evaluar clustering
        print("Evaluando clustering...")
        clustering_results = self.evaluate_clustering(embeddings)
        
        # 6. Visualizar clusters
        print("Visualizando clusters...")
        cluster_labels = self.visualize_clusters(embeddings, n_clusters, tsne_results)
        
        # 7. Evaluar calidad de embeddings
        print("Evaluando calidad de embeddings...")
        quality_metrics = self.evaluate_embedding_quality(embeddings)
        
        # 8. Evaluar recuperación de imágenes similares
        retrieval_results = None
        if query_indices is None:
            # Seleccionar aleatoriamente algunas imágenes como consultas
            n_queries = min(5, len(image_paths))
            query_indices = np.random.choice(len(image_paths), size=n_queries, replace=False)
            
        print(f"Evaluando recuperación de imágenes similares para {len(query_indices)} consultas...")
        retrieval_results = self.evaluate_retrieval(query_indices, top_k, embeddings, similarity_matrix)
        
        # 9. Visualizar resultados de recuperación
        print("Visualizando resultados de recuperación...")
        retrieval_vis_paths = self.visualize_retrieval_results(retrieval_results)
        
        # 10. Guardar resultados
        results = {
            'embeddings': embeddings.cpu().numpy() if save_results else None,
            'similarity_matrix': similarity_matrix if save_results else None,
            'tsne_results': tsne_results,
            'cluster_labels': cluster_labels,
            'clustering_metrics': clustering_results.to_dict() if isinstance(clustering_results, pd.DataFrame) else clustering_results,
            'quality_metrics': quality_metrics,
            'retrieval_results': retrieval_results,
            'visualization_paths': {
                'similarity_matrix': str(sim_matrix_path),
                'tsne': str(self.output_dir / "tsne_visualization.png"),
                'clustering': str(self.output_dir / "clustering_evaluation.png"),
                'clusters': str(self.output_dir / f"clusters_visualization_k{n_clusters}.png"),
                'retrieval': [str(p) for p in retrieval_vis_paths]
            }
        }
        
        # Guardar resultados como JSON si se solicita
        if save_results:
            import json
            
            # Filtrar resultados que no son serializables
            serializable_results = {
                'clustering_metrics': results['clustering_metrics'],
                'quality_metrics': results['quality_metrics'],
                'visualization_paths': results['visualization_paths']
            }
            
            with open(self.output_dir / "evaluation_results.json", 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            print(f"Resultados guardados en {self.output_dir / 'evaluation_results.json'}")
        
        print("Evaluación completa finalizada.")
        return results
