# Conditional VAEs and Normalising Flows

## Conditional VAEs

C-VAEs improve traditional VAEs by taking additional inputs, or "conditions". This condition can be any piece of relevant information about the data that we want the VAE to take into account, allowing for a greater degree of control over the outputs. 

In this case we permit the VAE to view the ground truth age, ethnicity and gender labels of the images. This allows the model to create more accurate reconstructions. 

### Reconstructions

![image](https://github.com/alexjmanlove/convolutional-variational-autoencoders/assets/79708390/46e08fee-5bba-4d3a-817b-5de660e3579e)
![image](https://github.com/alexjmanlove/convolutional-variational-autoencoders/assets/79708390/cb18747b-8a49-421c-b16e-b3c3f7aa194f)
![image](https://github.com/alexjmanlove/convolutional-variational-autoencoders/assets/79708390/a27e86ae-3675-4711-97ca-0e80ed3b2ed2)
![image](https://github.com/alexjmanlove/convolutional-variational-autoencoders/assets/79708390/2d9689b9-a4e9-4fd4-8273-5a85dd50e370)


### TODO 

- Sample from the latent space for suggested input classes.
- Fix the features and vary the class labels to interpolate along age, gender, ethnicity etc. 
