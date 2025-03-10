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

The dataset is designed for research in **AI-generated art analysis**, **image similarity**, and **style transfer**, as well as for training models to compare human-created and AI-generated images.

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
| `original_image` | Path to the original painting from WikiArt. |
| `prompt_complete` | Full prompt generated using LLaMA 3 8B. |
| `prompt_compressed` | Compressed version of the prompt optimized for Stable Diffusion XL Refiner 1.0 (max 75 tokens). |
| `generated_image` | Path to the AI-generated image. |

---

## Artistic Genres Included
The dataset spans **27 artistic genres**, with the following distribution:

| Genre | Image Count |
|----------------------------|-------------|
| Impressionism | 13,028 |
| Realism | 10,546 |
| Romanticism | 6,919 |
| Expressionism | 6,335 |
| Post Impressionism | 6,307 |
| Symbolism | 4,524 |
| Baroque | 4,236 |
| Art Nouveau Modern | 4,168 |
| Abstract Expressionism | 2,594 |
| Northern Renaissance | 2,551 |
| Naive Art Primitivism | 2,385 |
| Cubism | 2,177 |
| Rococo | 2,087 |
| Color Field Painting | 1,567 |
| Pop Art | 1,483 |
| Early Renaissance | 1,389 |
| High Renaissance | 1,341 |
| Minimalism | 1,328 |
| Mannerism Late Renaissance | 1,277 |
| Ukiyo-e | 1,163 |
| Fauvism | 923 |
| Pointillism | 501 |
| Contemporary Realism | 481 |
| New Realism | 313 |
| Synthetic Cubism | 216 |
| Analytical Cubism | 110 |
| Action Painting | 93 |

---

## Prompt Generation Process
Prompts were dynamically created using **LLaMA 3 8B** based on the original painting's metadata (artist, title, genre, and description). The prompts were designed to guide Stable Diffusion XL Refiner 1.0 in modifying the images while maintaining or transforming specific artistic elements.

### Modification Levels
1. **Moderate Modification**:
   - Preserves composition and subject.
   - Adjusts color schemes, lighting, or artistic techniques.
   - Introduces secondary elements.

2. **Radical Modification**:
   - Transforms style, era, or medium.
   - Alters color palette, composition, or perspective.
   - Reinterprets subject matter conceptually.

To ensure compatibility with Stable Diffusion XL Refiner 1.0, prompts were compressed to a maximum of **75 tokens** using **facebook/bart-large-cnn**.

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
