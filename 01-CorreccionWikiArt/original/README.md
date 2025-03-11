---
tags:
- WikiArt
- art
- AI-generated
- image-to-image
- Stable-Diffusion
- SDXL
- painting
- digital-art
- similarity
- computer-vision
- dataset
- image-processing
- fine-tuning
size_categories:
- 10K<n<100K
---
# WikiArt Enhanced Dataset

## Description
This dataset contains 81,444 artistic images from WikiArt, organized into different artistic genres. It has undergone several improvements and corrections to optimize its use in machine learning tasks and computational art analysis. Credits to the original author of daset go to: [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart/data?select=Art_Nouveau_Modern)

## Enhancements

### 1. Encoding Issues Correction
- Fixed encoding issues in filenames and artist information.
- All filenames were renamed for consistency.
- Artist names were normalized, removing incorrect characters or encoding errors.

### 2. Genre Normalization
- Initially, some images belonged to multiple genres. The classification was reduced to a single main genre per image.
- Genres were consolidated into 27 main categories:

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

- Genres were converted from list format (`['Abstract Expressionism']`) to string (`"Abstract Expressionism"`).

### 3. Data Cleaning and Restructuring
- The `description` column was renamed to `painting_name`, removing hyphens for better readability.
- The columns `width`, `height`, and `genre_count` were removed.
- Missing values in `phash`, `subset`, `genre`, and `artist` were filled (2097 affected rows).
- `phash` values were generated for missing images.

### 4. Subset Reassignment
- The original distribution was:
  - Train: 63,998
  - Test: 16,000
  - Uncertain artist: 44
- A stratified 80/20 split was applied:
  - Train: 65,155
  - Test: 16,289

### 5. Automatic Description Generation
- A new `description` column was added, with descriptions generated using the [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- Example of the final dataset entry:

```
file_name,genre,artist,painting_name,phash,description,subset
Impressionism/william-merritt-chase_still-life-with-cockatoo.jpg,Impressionism,william merritt chase,still life with cockatoo,b0e24b85961e6de9,a painting of a white bird sitting on a vase,train
```

## Final Dataset Format
The dataset is structured in a CSV file with the following columns:

- **file_name**: Image filename.
- **genre**: Assigned artistic genre.
- **artist**: Artist's name.
- **painting_name**: Painting title, cleaned and formatted.
- **phash**: Perceptual hash of the image.
- **description**: Automatically generated image description.
- **subset**: Subset to which the image belongs (train or test).

## Usage
This dataset can be used for art classification tasks, automatic description generation, artistic style analysis, and training vision models.

## License
This dataset is based on WikiArt and is intended for academic and research purposes.

