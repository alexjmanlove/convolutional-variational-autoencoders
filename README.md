# Convolutional Variational Autoencoders for Generative Modelling of Human Faces with `PyTorch`

In this project we apply convolutional neural networks in variational autoencoder models to perform unsupervised reconstruction, damage restoration and interpolation of human face images.

# Project Preview

## What is a Convolutional Neural Network?

In this project we start by briefly recapping the convolutional neural networks and explore their use for supervised tasks of classification and regression.   
![convolutions-1](https://user-images.githubusercontent.com/79708390/212490621-2c4ed11c-b6fb-4888-a073-bae63c1f08de.png)

## What is a Variational Autoencoder?

Secondly, we briefly recap the variational autoencoder, which builds on the approach of traditional autoencoders by incorporating principles of probabilistic generative modelling and Bayesian variational inference to learn a distribution over the possible latent codes. In particular, here we use convolutional encoders and deconvolutional decoders.

<img src="https://user-images.githubusercontent.com/79708390/211933944-f558ba34-7042-4e96-9f0a-ff16ce8605f9.png" width="900" height="480">

## How Do `latent_dimension` and `β` Regularisation Affect VAE Reconstruction Quality?

The number of permitted latent dimensions is a hyperparameter that determines the size of the latent space. This hyperparameter presents a bias-variance tradeoff. The β coefficient is a hyperparameter introduced in the loss function that controls the strength of regularisation. We explore how different values of these hyperparameters affect the resulting reconstructions produced by our VAEs.

![all_vae_models_reconstructions-1](https://user-images.githubusercontent.com/79708390/211932692-e81312ef-b85e-4e9c-9c2b-f385a4ca5ece.png)

## Face Interpolation Matirx Sampled From Latent Space

Using these models, we can generate new images that are similar to the training data by sampling from the learned latent distribution. 

<img src="https://user-images.githubusercontent.com/79708390/211932228-ea829d3a-4c48-41e7-9a5e-275f42ae4c64.png" width="600" height="600">

## Denoising and Damage Restoration of Corrupted Images

Here show some examples using VAEs for denoising and damage restoration. 

### Simple Mean Filter Denoising vs VAE Reconstruction

Using the VAE we can create intelligible reconstructions of noisy images, assuming the original image is sufficiently similar to the training data. In the case of severe Gaussian noise, we can see the results of the VAE reconstruction are better than other simpler methods like mean filter.

<img src="https://user-images.githubusercontent.com/79708390/211933261-4777d142-448b-4654-8897-27721df0e73f.png" width="380" height="470">

### Inpainting Deleted Data

Here, the VAE model is able to reconstruct an approximation of the original images, effectively in-painting the deleted data. However we note that the dataset was biased, with many observations belonging to ethnicity class 0, i.e. white caucasian. As a result the reconstructions of images belonging to minority classes can be of low fidelity. In particular the individual in the center column appears to have changed ethnicity, which exemplifies the importance of ML ethics.

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
