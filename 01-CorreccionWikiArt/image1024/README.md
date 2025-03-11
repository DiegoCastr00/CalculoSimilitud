
---
tags:
- WikiArt
- art
- image-to-image
- Stable-Diffusion
- painting
- digital-art
- computer-vision
- dataset
- LANCZOS
- image-processing
- resized-images
- 1024x1024
- deep-learning
- convolutional-neural-networks
- feature-extraction
- image-classification
- machine-learning
size_categories:
- 10K<n<100K
---
# WikiArt Resized Dataset

## Description
This dataset contains 81,444 artistic images from WikiArt, organized into different artistic genres. The images have been resized to a uniform resolution of 1024x1024 pixels using LANCZOS resampling, ensuring consistency for machine learning tasks and computational art analysis. The base for the dataset was [Dant33/WikiArt-81K-BLIP_2-captions](https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-captions) 

## Enhancements

### 1. Image Resizing
- All images have been resized to **1024x1024 pixels**.
- The resizing was performed using `target_size=(1024, 1024), resample=Image.LANCZOS` to maintain quality and minimize artifacts.
- The filenames were updated to reflect this change by appending `_resize1024` to the original filename.

### 2. Genre Normalization
- Each image is classified under a **single** main artistic genre.
- The dataset contains **27 main genres**:

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

### 3. Automatic Description Generation
- Each image includes a textual description generated using the [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b) model.

## Dataset Format
The dataset is structured in a CSV file with the following columns:

- **file_name**: Image filename, including the `_resize1024` suffix.
- **genre**: Assigned artistic genre.
- **artist**: Artist's name.
- **painting_name**: Painting title, cleaned and formatted.
- **phash**: Perceptual hash of the image.
- **description**: Automatically generated image description.
- **subset**: Subset to which the image belongs (train or test).

### Example Entry
```
file_name,genre,artist,painting_name,phash,description,subset
Impressionism/william-merritt-chase_still-life-with-cockatoo_resize1024.jpg,Impressionism,william merritt chase,still life with cockatoo,b0e24b85961e6de9,a painting of a white bird sitting on a vase,train
```

## Usage
This dataset is ideal for:
- Artistic style classification.
- Training deep learning models for image-based tasks.
- Automatic art description generation.
- Image retrieval and similarity analysis.