# Deep Generative Modelling with Convolutional Variational Autoencoders

In this project we apply convolutional neural networks in variational autoencoder models to perform unsupervised reconstruction, damage restoration and interpolation of human face images.

# Project Preview

## What is a Variational Autoencoder?

<img src="https://user-images.githubusercontent.com/79708390/211933944-f558ba34-7042-4e96-9f0a-ff16ce8605f9.png" width="900" height="480">

## How Do `latent_dimension` and `β` Regularisation Affect VAE Reconstruction Quality?

![all_vae_models_reconstructions-1](https://user-images.githubusercontent.com/79708390/211932692-e81312ef-b85e-4e9c-9c2b-f385a4ca5ece.png)

## Face Interpolation Matirx Sampled From Latent Space

<img src="https://user-images.githubusercontent.com/79708390/211932228-ea829d3a-4c48-41e7-9a5e-275f42ae4c64.png" width="600" height="600">

## Denoising and Damage Restoration of Corrupted Images

### Simple Mean Filter Denoising vs VAE Reconstruction

<img src="https://user-images.githubusercontent.com/79708390/211933261-4777d142-448b-4654-8897-27721df0e73f.png" width="380" height="470">

### Inpainting Deleted Data

<img src="https://user-images.githubusercontent.com/79708390/211933289-aec02791-0020-41ed-9727-5ad5fc6dd6b7.png" width="380" height="360">



## Reproducibility

The .ipynb file is organised into two chapters which loosely follow along with the PDF report. I recommend reading the full report first, then viewing the cell outputs in the notebook, before you try to run the code yourself.     

*N.B.: This project involved training three convolutional neural networks and four variational autoencoders. These models were trained on GPU and collectively took over an hour of compute time.*

### Python Package Requirements

An outline of the packages used in the project is available in the requirements.txt. In particular, the following packages:
- numpy>=1.21.6
- pandas>=1.3.5
- matplotlib>=3.2.2
- seaborn>=0.11.2
- scikit-learn>=1.0.2
- torch>=1.13.0+cu116
- torchvision>=0.14.0+cu116
- piqa>=1.2.2
- pyarrow>=9.0.0
