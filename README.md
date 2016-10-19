# robust-classification
Deep Learning Project

In this project, we designed robust deep learning techniques for classifying images that have different levels of Gaussian noise or Instagram filters applied.

A neural network designed and trained on clean images would perform very poorly when classifying images that have had some noise applied. We implemented Denoising Autoencoders (DAEs) to restore corrupted images to closely resemble the originals and achieved 30-40% of accuracy improvement when feeding the outputs from DAEs to traditional Convolutional Neural Network (CNN) for classification.
