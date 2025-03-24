# 🎨 Similitud entre Imágenes Artísticas y Generadas por Stable Diffusion
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Descripción

Este proyecto investiga y cuantifica la similitud entre obras artísticas originales de WikiArt y sus versiones generadas mediante Stable Diffusion XL Refiner 1.0, utilizando redes convolucionales siamesas basadas en CLIP y aprendizaje contrastivo.

### Motivación

Con el auge de modelos generativos como Stable Diffusion, resulta crucial entender y medir objetivamente qué tan similares son las imágenes generadas por IA comparadas con obras artísticas originales. Este trabajo proporciona una metodología y herramientas para cuantificar estas similitudes.

## 🚀 Principales Características

- Dataset curado de 81,444 imágenes de WikiArt con sus correspondientes versiones generadas por SDXL
- Generación de descripciones automáticas con BLIP2
- Prompts creativos generados con LLaMA 3 8B
- Modelo siamés basado en CLIP para cálculo de similitud
- Análisis comparativo entre arte original y generado por IA

## 📊 Resultados

Algunos ejemplos de comparaciones entre obras originales y generadas:

<div align="center">
    <figure>
        <img 
            src="https://raw.githubusercontent.com/DiegoCastr00/CalculoSimilitud/refs/heads/master/07-SIamesCLIP/inference/output.png" 
            width="700" 
            alt="Comparación de Similitud 1"
        />
    </figure>
</div>


<div align="center">
    <figure>
        <img 
            src="https://raw.githubusercontent.com/DiegoCastr00/CalculoSimilitud/refs/heads/master/07-SIamesCLIP/inference/output1.png" 
            width="700" 
        />
    </figure>
</div>


<div align="center">
    <figure>
        <img 
            src="https://raw.githubusercontent.com/DiegoCastr00/CalculoSimilitud/refs/heads/master/07-SIamesCLIP/inference/output2.png" 
            width="700" 
            alt="Example"
        />
    </figure>
</div>

## 🛠️ Componentes del Proyecto

### Datasets

- **Dataset corregido:** [WikiArt-81K-BLIP_2-captions](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-captions)
- **Dataset con resize a 768x768:** [WikiArt-81K-BLIP_2-768x768](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-768x768)
- **Dataset con resize a 1024x1024:** [WikiArt-81K-BLIP_2-1024x1024](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-1024x1024)
- **Dataset generado:** [Wikiart_with_StableDiffusion](https://huggingface.co/datasets/Dant33/Wikiart_with_StableDiffusion)

### Modelos y Técnicas

- **Corrección de WikiArt**: Normalización de géneros, corrección de codificación y limpieza de datos
- **Resize de Imágenes**: Interpolación Lanczos y padding negro para preservar proporciones
- **Descripciones BLIP2**: Generadas con `Salesforce/blip2-opt-2.7b`
- **Prompts LLaMA 3**: Generados con `meta-llama/Meta-Llama-3-8B-Instruct` y comprimidos con `facebook/bart-large-cnn`
- **Generación SDXL**: Stable Diffusion XL Refiner 1.0 con optimizaciones (steps: 25, strength: 0.4, guidance: 7.5)
- **Modelo Siamés CLIP**: Arquitectura contrastiva para evaluación de similitud


## 📦 Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas:

- **00-huggingface**: Scripts para subir datasets/modelos a Hugging Face
- **01-CorreccionWikiArt**: Dataset WikiArt corregido (original y versiones redimensionadas)
- **02-Resize**: Códigos para el procesamiento y redimensionado de imágenes
- **03-Descripciones**: Generación de descripciones con BLIP2
- **04-Prompts**: Generación de prompts con LLaMA 3 8B
- **05-SDXL**: Procesador para crear imágenes con SDXL y logs de generación
- **06-DescripcionesSDXL**: Descripciones de imágenes generadas usando BLIP2
- **07-SiamesCLIP**: Modelo siamés basado en CLIP para cálculo de similitud


## 🔧 Instalación y Uso
Instalar dependencias con:
```bash
pip install -r requirements.txt
```

## 📧 Contacto

Si tienes preguntas o sugerencias, contáctame a través de [GitHub Issues](https://github.com/DiegoCastr00/CalculoSimilitud/issues) o [email](mailto:diego.castro.elvira@gmail.com).
