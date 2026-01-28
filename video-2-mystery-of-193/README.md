# Video 2: Dissecting Gemma 3 Image Tokenization: The Mystery of 193

Frozen artifact for the [YouTube video](https://youtu.be/3FMOknkH9XM). **This code will never change.**

## Notebooks

The notebooks tell a research story in sequence:

1. **`0_dataset_context.ipynb`** - Dataset setup and context for the investigation
2. **`1_finding_outlier_positions.ipynb`** - Discovery that 95% of images have position 193 as a representational outlier
3. **`2_outlier_positions_under_rotation.ipynb`** - Testing rotation invariance (the outlier is not about spatial position)
4. **`3_cross_image_outlier_similarity.ipynb`** - Position 193 embeddings are highly consistent across images (0.912 similarity)
5. **`4_semantic_meaning_in_outliers.ipynb`** - Ablation and steering experiments (zeroing doesn't break the model, but flipping direction does)

## Files

- **`vlm_tools.py`** - Shared utility functions for model loading, similarity analysis, and steering experiments
- **`images/`** - Test images including rotation variants
- **`open_images/`** - Open Images dataset utilities for embedding database population
