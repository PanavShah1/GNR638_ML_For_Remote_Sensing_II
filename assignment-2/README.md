# Image Classification using MLP Networks

This assignment compares two approaches for image classification using Multi-Layer Perceptron (MLP) networks:

1. **Bag-of-Visual-Words (BoVW) Approach**:  
   In this approach, SIFT features are extracted from the images and clustered using K-Means to create a visual vocabulary. Each image is then represented as a normalized histogram of visual words (BoVW representation), which is used as input to an MLP classifier. This approach is implemented in `main.py`.

2. **Flattened Image Approach**:  
   Each grayscale image is resized (to 72x72), flattened into a one-dimensional vector, and then fed directly to an MLP network. This approach is implemented in `main2.py`.

## Project Structure

- **main.py**

  - Extracts SIFT features from images.
  - Trains or loads a K-Means visual vocabulary.
  - Computes BoVW representations for training, validation, and test sets.
  - Trains an MLP classifier on the BoVW features.
  - Saves classification outputs and a visualization in the `output/` directory.

- **main2.py**

  - Resizes images to 72x72, then flattens each image.
  - Trains an MLP classifier directly on the flattened images.
  - Saves the classification results in the `output/` directory.

- **cache/**  
  This directory stores intermediate data (feature caches, trained models, etc.) for reproducibility.

- **output/**  
  This directory contains the output results including accuracy reports and visualizations.

## Results

- **BoVW Approach:**

  - Validation Accuracy: 46.19%
  - Test Accuracy: 41.43%

- **Flattened Image Approach:**
  - Test Accuracy: 5.24%

The superior performance of the BoVW approach demonstrates the value of extracting meaningful features from images before classification, in contrast to using raw flattened data.

## Conclusion

This assignment illustrates the importance of feature engineering in image classification tasks. While raw pixel intensities offer one method of input, employing a Bag-of-Visual-Words representation allows the network to leverage richer, more discriminative features, leading to significantly better classification performance.
