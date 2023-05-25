# Conditional VAEs and Normalising Flows

## Conditional VAEs

C-VAEs improve traditional VAEs by taking additional inputs, or "conditions". This condition can be any piece of relevant information about the data that we want the VAE to take into account, allowing for a greater degree of control over the outputs. 

In this case we permit the VAE to view the ground truth age, ethnicity and gender labels of the images. This allows the model to create more accurate reconstructions. 

### Reconstructions

By providing class labels we can improve the quality and fidelity of the reconstructions.

![image](https://github.com/alexjmanlove/convolutional-variational-autoencoders/assets/79708390/0a3b8a98-9697-428e-b57b-4c25f7742fff)


### TODO 

- Sample from the latent space for suggested input classes.
- Fix the features and vary the class labels to interpolate along age, gender, ethnicity etc. 
- Normalising Flows 
