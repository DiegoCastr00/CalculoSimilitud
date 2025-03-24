# Cálculo de Similitud entre Imágenes Artísticas y Generadas por Stable Diffusion utilizando Redes Convolucionales Siamesas

Este repositorio contiene el código y los datos utilizados en el proyecto de investigación sobre la similitud entre imágenes artísticas y generadas mediante Stable Diffusion XL Refiner 1.0, utilizando redes convolucionales siamesas basadas en CLIP.

## Estructura del Repositorio

```
00-huggingface/      # Scripts para comprimir y subir datasets/modelos a Hugging Face
01-CorreccionWikiArt/ # Dataset corregido (original, resize 1024, resize 768)
02-Resize/           # Códigos utilizados para el resize de imágenes
03-Descripciones/    # Generación de descripciones con BLIP2
04-Prompts/          # Generación de prompts con LLaMA 3 8B
05-SDXL/             # Procesador para crear imágenes con SDXL y logs
06-DescripcionesSDXL/ # Descripciones de imágenes generadas con SDXL usando BLIP2
07-SIamesCLIP/       # Modelo siamese basado en CLIP para similitud de imágenes
imagenes/            # Imágenes de resultados y comparaciones

.gitignore
README.md
context.md
requirements.txt
```

---

## 00 - Hugging Face Upload Scripts
Scripts utilizados para la compresión y subida de archivos a Hugging Face, tanto para los datasets como para modelos entrenados.

---

## 01 - Corrección del Dataset WikiArt
El dataset original de WikiArt fue corregido y mejorado para este estudio. Se encuentran disponibles tres versiones:

- **Dataset corregido:** [WikiArt-81K-BLIP_2-captions](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-captions)
- **Dataset con resize a 768x768:** [WikiArt-81K-BLIP_2-768x768](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-768x768)
- **Dataset con resize a 1024x1024:** [WikiArt-81K-BLIP_2-1024x1024](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-1024x1024)

Se realizaron las siguientes mejoras:
1. Corrección de problemas de codificación.
2. Normalización de géneros artísticos.
3. Limpieza y reestructuración de datos.
4. Generación de descripciones automáticas con BLIP2.

Cada subdirectorio (original, image1024, image768) contiene su propio README con más detalles.

---

## 02 - Resize de Imágenes
Se utilizó interpolación Lanczos para asegurar la calidad de las imágenes:
```python
resize(target_size=(1024, 1024), resample=Image.LANCZOS)
```
Se añadió **padding negro** en 768x768 para conservar la proporción original.

---

## 03 - Generación de Descripciones con BLIP2
Se utilizaron los siguientes modelos:
```python
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
```
Las descripciones de mayor calidad están en `01-CorreccionWikiArt`.

---

## 04 - Generación de Prompts con LLaMA 3 8B
Prompts generados con:
- **Modelo base:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Compresión de prompts:** `facebook/bart-large-cnn` (hasta 75 tokens)

### Tipos de Modificación en los Prompts:
- **Modificación Moderada:** Se mantiene la composición y tema original, pero con técnicas nuevas.
- **Modificación Radical:** Transformación completa en estilo, medio o concepto.

---

## 05 - Generación de Imágenes con Stable Diffusion XL
Se usó **Stable Diffusion XL Refiner 1.0** con la siguiente configuración:
- **Modelo:** `stabilityai/stable-diffusion-xl-refiner-1.0`
- **VAE:** `madebyollin/taesdxl`
- **Precisión:** FP16
- **Optimizaciones:** GPU acceleration, VAE Slicing, Xformers Attention
- **Parámetros de inferencia:**
  - **Steps:** 25
  - **Strength:** 0.4
  - **Guidance Scale:** 7.5
  - **Batch Size:** 6

Dataset generado: [Wikiart_with_StableDiffusion](https://huggingface.co/datasets/Dant33/Wikiart_with_StableDiffusion)

---

## 06 - Descripciones de Imágenes Generadas con SDXL
Se generaron con **BLIP2** siguiendo el mismo proceso de `03-Descripciones`.

---

## 07 - Modelo Siamesa Basado en CLIP
Incluye:
- **Entrenamiento del modelo** con aprendizaje contrastivo por tripletes.
- **Inferencia y evaluación de similitud** entre imágenes artísticas y generadas.
- **Métricas:** MSE, MAE, Cosine Similarity.

Imágenes de salida disponibles en `07-SIamesCLIP/inference/`.

---

## Resultados
Algunos ejemplos de comparaciones y resultados:

<div align="center">
    <figure>
        <img 
            src="https://raw.githubusercontent.com/DiegoCastr00/CalculoSimilitud/refs/heads/master/07-SIamesCLIP/inference/output.png" 
            width="600" 
            alt="Example"
        />
    </figure>
</div>

---

## Requerimientos
Instalar dependencias con:
```bash
pip install -r requirements.txt
```

## Contacto
Si tienes preguntas o sugerencias, contáctame en GitHub.

---

Este README proporciona una visión general del repositorio. Si necesitas información detallada, revisa los README dentro de cada carpeta.

