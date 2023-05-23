# Conditional VAEs and Normalising Flows

## Conditional VAEs

C-VAEs improve traditional VAEs by taking additional inputs, or "conditions". This condition can be any piece of relevant information about the data that we want the VAE to take into account, allowing for a greater degree of control over the outputs. 

In this case we permit the VAE to view the ground truth age, ethnicity and gender labels of the images. This allows the model to create more accurate reconstructions. 