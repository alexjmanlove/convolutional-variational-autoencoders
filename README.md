# Convolutional Variational Autoencoders

![myfaces](https://user-images.githubusercontent.com/79708390/229932467-587beba8-9e70-40dc-a48f-2f1135778927.png)

This is a collection of projects using both `PyTorch` and `TensorFlow` to implement convolutional neural networks for classification and regression as well as convolutional variational autoencoders for generative image modelling.

## Project 1

**Chapter 1** firstly demonstrates the use of a simple convolutional neural network for supervised tasks: gender classification, ethnicity classification, and age regression. Secondly, we use variational autoencoders to generate novel images.

This image is a previews how changing the hyperparameter values of latent dimension size and $\beta$ regularisation coefficient affect reconstruction quality.
![all_vae_models_reconstructions-1](https://user-images.githubusercontent.com/79708390/229934163-b6b4dfa9-c1e5-4214-9976-c6f49f9e4cd5.png)

This is an interpolation matrix sampled purely from the latent space. None of these folks are real!
<img src="https://user-images.githubusercontent.com/79708390/229935641-4c33c68b-b5ab-463c-985a-00e05b75573c.png" height=800, width=800>


## Project 2

**Chapter 2** explores the use of conditional and normalising flows to improve the flexibility and fidelity of the VAE reconstructions.
