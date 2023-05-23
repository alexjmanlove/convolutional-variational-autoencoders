# Convolutional Variational Autoencoders

![myfaces](https://user-images.githubusercontent.com/79708390/229932467-587beba8-9e70-40dc-a48f-2f1135778927.png)

This is a collection of projects using both `PyTorch` and `TensorFlow` to implement convolutional neural networks for classification and regression as well as convolutional variational autoencoders for generative image modelling.

## Project 1 using `PyTorch`
Firstly we demonstate the use simple convolutional neural networks for supervised tasks: gender classification, ethnicity classification, and age regression. Secondly, we use variational autoencoders to generate novel images.

This image previews how the effect varying hyperparameters of latent dimension size and $\beta$ regularisation coefficient affect reconstruction quality.
![all_vae_models_reconstructions-1](https://user-images.githubusercontent.com/79708390/229934163-b6b4dfa9-c1e5-4214-9976-c6f49f9e4cd5.png)

### Face Interpolation Matrix
In this figure we apply the trained model to generate a face interpolation matrix. Firstly, we generate the four corner faces by decoding samples $\mathcal{Z} = \{z_1, z_2, z_3, z_4\}$ drawn directly from the learned latent distribution. In the latent space, we perform linear interpolation between pairs of these four latent vectors. This is performed  by formulating the linear interpolation function, where given two latent vectors $z_i$ and $z_j$, we have $f(\lambda; z_i, z_j) = (1-\lambda) z_i + \lambda z_j, \hspace{.33em} \lambda \in [0,1].$ We vary $\lambda$ and build up a collection of latent vectors, taken at regular intervals from along the line segments associated to this function. This collection of latent vectors is decoded and visualised below in Fig. \ref{fig:generation}. Note that no restriction was placed on the initial samples $\mathcal{Z}$, meaning $z_1, ..., z_4$ are not equidistant from one another.
<img src="https://user-images.githubusercontent.com/79708390/229935641-4c33c68b-b5ab-463c-985a-00e05b75573c.png" height=800, width=800>


## Project 2 using `TensorFlow`

<img height=200 width=100 src="https://github.com/alexjmanlove/convolutional-variational-autoencoders/assets/79708390/26dabcb4-0ca9-4d9c-a508-9852e4c007ac">

Secondly, we explore the use of labelled inputs for conditional VAEs as well as normalising flows to improve the flexibility and fidelity of the VAE reconstructions.



#### Status
- Conditional VAE: `[Implemented]`
- VAE IAF: `[Coming soon]`
