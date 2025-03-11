---
tags:
- art
- AI-generated
- image-to-image
- Stable-Diffusion
- SDXL
- painting
- digital-art
- deep-learning
- similarity
- convolutional-neural-networks
- siamese-networks
- computer-vision
- dataset
- image-processing
- fine-tuning
size_categories:
- 10K<n<100K
---

# Artistic Images Transformed by Stable Diffusion XL Refiner 1.0

## Overview
This dataset contains **81,444 AI-generated images** derived from famous paintings across **27 artistic genres**. The transformation process involved resizing the original images to 768px, generating detailed descriptions using **BLIP2**, and creating customized prompts with **LLaMA 3 8B**. These prompts were then used with **Stable Diffusion XL Refiner 1.0** to generate modified versions of the original artworks.

The dataset is designed for research in **AI-generated art analysis**, **image similarity**, and **style transfer**, as well as for training models to compare human-created and AI-generated images. Unlike traditional datasets, este conjunto de datos permite analizar cómo los modelos de IA pueden reinterpretar pinturas clásicas, aplicando transformaciones estilísticas y estructurales basadas en descripciones generadas automáticamente.

---
## Original Dataset
The original dataset used for this project was sourced from [WikiArt Resize](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-768x768), a comprehensive online repository of artworks. You can explore the original paintings and their metadata [Wikiart](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-captions).

---

## Dataset Structure

### Directory Structure
The images are organized by artistic genre in the following format:
```
images/{artistic_genre}/{original_painting_name}_resize768_generated
```
For example:
```
images/Minimalism/yves-klein_untitled-blue-monochrome-1956_resize768_generated
```
where:
- `Minimalism` represents the **artistic genre**.
- `yves-klein_untitled-blue-monochrome-1956` represents the **original painting's name**.
- `_resize768_generated` indicates that the image was resized to 768px before being processed by Stable Diffusion.


### Metadata Files
#### `data.csv`
Contains two columns:
| Column | Description |
|---------|-------------|
| `generated_image` | Path to the AI-generated image (e.g., `Impressionism/pierre-auguste-renoir_...`). |
| `prompt` | The prompt used to generate the image with Stable Diffusion XL Refiner 1.0. |

#### `metadata.csv`
Provides extended metadata:
| Column | Description |
|---------|-------------|
| `original_image` | Path to the original painting from [WikiArt Resize](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-768x768). |
| `prompt_complete` | Full prompt generated using LLaMA 3 8B. |
| `prompt_compressed` | Compressed version of the prompt optimized for Stable Diffusion XL Refiner 1.0 (max 75 tokens). |
| `generated_image` | Path to the AI-generated image. |

---

## Artistic Genres Included
The dataset spans **27 artistic genres**, with the following distribution:

| #  | Genre                        | Size  |
|----|------------------------------|-------|
| 1  | Impressionism                | 13060 |
| 2  | Realism                      | 10733 |
| 3  | Romanticism                  | 7019  |
| 4  | Expressionism                | 6736  |
| 5  | Post Impressionism           | 6450  |
| 6  | Symbolism                    | 4528  |
| 7  | Art Nouveau Modern           | 4334  |
| 8  | Baroque                      | 4240  |
| 9  | Abstract Expressionism       | 2782  |
| 10 | Northern Renaissance         | 2552  |
| 11 | Naive Art Primitivism        | 2405  |
| 12 | Cubism                       | 2235  |
| 13 | Rococo                       | 2089  |
| 14 | Color Field Painting         | 1615  |
| 15 | Pop Art                      | 1483  |
| 16 | Early Renaissance            | 1391  |
| 17 | High Renaissance             | 1343  |
| 18 | Minimalism                   | 1337  |
| 19 | Mannerism Late Renaissance   | 1279  |
| 20 | Ukiyo e                      | 1167  |
| 21 | Fauvism                      | 934   |
| 22 | Pointillism                  | 513   |
| 23 | Contemporary Realism         | 481   |
| 24 | New Realism                  | 314   |
| 25 | Synthetic Cubism             | 216   |
| 26 | Analytical Cubism            | 110   |
| 27 | Action painting              | 98    |

---

## Prompt Generation Process
Prompts were not simply generated but **carefully constructed** using **LLaMA 3 8B**, leveraging metadata from the original painting (artist, title, genre, and a BLIP2-generated description). The purpose of these prompts was to encourage Stable Diffusion XL Refiner 1.0 to modify the images while either maintaining core artistic features or transforming them in specific ways.

### **Types of Prompt Modifications**
1. **Moderate Modification**:
   - Preservation of the original composition and subject.
   - Introduction of new artistic techniques, alternative lighting schemes, or subtle transformations in color palettes.

2. **Radical Modification**:
   - Complete re-interpretation of the artwork in a different artistic style.
   - Change of medium (e.g., oil painting to watercolor or digital art).
   - Conceptual transformation of elements within the scene.

To ensure compatibility with Stable Diffusion XL Refiner 1.0, prompts were compressed to a maximum of **75 tokens** using **facebook/bart-large-cnn**, ensuring optimal model performance and image quality consistency.

---
## Stable Diffusion XL Refiner 1.0 - Model Parameters  

The images in this dataset were generated using **Stable Diffusion XL Refiner 1.0** with a carefully optimized configuration to ensure quality and efficiency. The model was set up with the following parameters:

### **Model Setup**
- **Base Model:** [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- **VAE:** [madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl) (Tiny Autoencoder for SDXL)  
- **Precision:** `fp16` (16-bit floating point)  
- **Memory Optimizations:**
  - **GPU Acceleration:** Enabled (`torch.compile` for inference speed-up)  
  - **VAE Slicing:** Enabled (reduces VAE memory usage)  
  - **Xformers Attention:** Enabled (improves inference speed and memory efficiency)  

### **Inference Parameters**
- **Number of inference steps:** `25`  
- **Strength (image influence):** `0.4`  
- **Guidance Scale:** `7.5`  
- **Batch Size:** `6`  

---
## Potential Applications
This dataset can be used for:
- **AI-generated art analysis**: Studying how AI interprets and modifies artistic styles.
- **Image similarity research**: Training models to compare AI-generated and human-created images.
- **Style transfer and generative AI**: Developing models for artistic recognition or improving generative techniques.
- **Prompt engineering**: Understanding how structured prompts influence AI-generated outputs.

---
## Acknowledgments
We thank the creators of **BLIP2**, **LLaMA 3 8B**, and **Stable Diffusion XL Refiner 1.0** for their contributions to this project.

---