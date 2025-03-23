Este mensaje es solamente para poner en contexto de mi proyecto
Resumen del Proyecto: Cálculo de Similitud entre Imágenes Artísticas y Generadas por Stable Diffusion
Descripción General
El proyecto busca construir un modelo para calcular la similitud entre imágenes de pinturas artísticas originales de WikiArt y sus versiones generadas por Stable Diffusion XL Refiner 1.0, utilizando redes neuronales convolucionales siamesas y aprendizaje contrastivo.
Objetivos

- Comparar cuantitativamente la similitud entre imágenes artísticas originales y sus versiones generadas por IA
- Desarrollar un modelo capaz de medir estas similitudes sin supervisión explícita
- Evaluar el rendimiento utilizando métricas de regresión como MSE y MAE
  Dataset
- 81,444 imágenes generadas por Stable Diffusion XL Refiner 1.0 basadas en obras de WikiArt
- Pares de imágenes: original + generada
- Imágenes aleatorias de diferentes géneros para servir como ejemplos negativos
  Metodología Actual

1. Arquitectura Siamesa Contrastiva
   Se implementará una red neuronal siamesa basada en CLIP (clip-vit-base-patch32) utilizando aprendizaje contrastivo con tripletes:

```
Imagen original (ancla) → CLIP congelado → Embedding (512) → Capas transformadoras → Embedding refinado
Imagen generada (positiva) → CLIP congelado → Embedding (512) → Capas transformadoras → Embedding refinado
Imagen diferente (negativa) → CLIP congelado → Embedding (512) → Capas transformadoras → Embedding refinado
```

2. Características Principales

- **Modelo Base**: CLIP (clip-vit-base-patch32) preentrenado y congelado para extracción de características
- **Compartición de Pesos**: Las tres ramas de la red siamesa comparten exactamente los mismos pesos
- **Capas Transformadoras**: Adición de capas personalizadas post-CLIP para refinar los embeddings según la tarea específica
- **Aprendizaje No Supervisado**: No requiere etiquetas de similitud manuales

3. Función de Pérdida Contrastiva
   Se utilizará una función de pérdida contrastiva para entrenar el modelo:

```
loss = -log(exp(sim_pos/τ) / (exp(sim_pos/τ) + exp(sim_neg/τ)))
```

Donde:

- `sim_pos` es la similitud del coseno entre la imagen original y la generada
- `sim_neg` es la similitud del coseno entre la imagen original y la diferente
- `τ` (tau) es el parámetro de temperatura que controla la sensibilidad del modelo

4. Proceso de Entrenamiento
1. Los tripletes de imágenes (original, generada, diferente) se procesan a través de la red siamesa
1. Se calculan las similitudes del coseno entre los pares
1. Se aplica la pérdida contrastiva para maximizar la similitud entre pares genuinos y minimizar con negativos
1. Solo se actualizan los pesos de las capas transformadoras, manteniendo CLIP congelado
1. Métricas y evaluación:
   Triplet Accuracy: mide si el modelo distingue correctamente entre pares positivos y negativos
   Matriz de similitud: visualiza relaciones entre varias imágenes
1. Evaluación del Modelo
   Una vez entrenado, el modelo podrá:

- Recibir dos imágenes como entrada
- Producir un valor de similitud entre -1 y 1 (o normalizado entre 0 y 1)
  Ventajas del Enfoque
- Aprovecha el poder de CLIP como extractor de características visuales
- No requiere etiquetado manual de similitudes
- Aprende específicamente qué características son relevantes para la comparación entre arte original y generado
- Produce una métrica interpretable de similitud
