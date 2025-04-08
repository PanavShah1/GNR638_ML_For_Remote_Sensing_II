# Hierarchical Variational Autoencoder for Land Use Image Generation

This project implements a Hierarchical Variational Autoencoder (HVAE) trained on the UC Merced Land Use dataset to generate realistic land use images. The HVAE utilizes a two-level hierarchy in its latent space (`z1` and `z2`) to capture different levels of abstraction in the data.

## Model Architecture

The HVAE consists of the following main components:

1.  **Encoder `q(z1|x)`:**

    - Takes an input image `x`.
    - Uses several convolutional layers (`Conv2d`) followed by a fully connected layer (`Linear`).
    - Outputs the parameters (mean `q_z1_mean` and log-variance `q_z1_logvar`) of the approximate posterior distribution `q(z1|x)`.

2.  **Encoder `q(z2|z1)`:**

    - Takes a sample `z1` drawn from `q(z1|x)`.
    - Uses a Multi-Layer Perceptron (MLP) consisting of `Linear` layers and ReLU activations.
    - Outputs the parameters (mean `q_z2_mean` and log-variance `q_z2_logvar`) of the approximate posterior distribution `q(z2|z1)`.

3.  **Prior Network `p(z1|z2)`:**

    - Takes a sample `z2` drawn from `q(z2|z1)`.
    - Uses an MLP (`Linear` layers and ReLU activations).
    - Outputs the parameters (mean `p_z1_mean` and log-variance `p_z1_logvar`) of the prior distribution `p(z1|z2)`. This network learns the relationship between the two latent levels.

4.  **Decoder `p(x|z1)`:**
    - Takes a sample `z1`.
    - Uses a fully connected layer followed by several transposed convolutional layers (`ConvTranspose2d`) with ReLU activations.
    - The final layer uses a Sigmoid activation to output the reconstructed image `recon_x`, with pixel values in the range [0, 1].

The latent dimensions used are `LATENT_DIM_1 = 128` and `LATENT_DIM_2 = 64`.

## Example Outputs

- **Generated Samples:** `output/hvae/samples/sample_50.png`

![set of 64 samples generated after 50 epochs](assets/sample_50.png)
