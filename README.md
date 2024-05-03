# Multi Layer Perceptron Numpy

Numpy-MLP is a multi-layer perceptron (MLP) library implemented purely in Numpy. This is designed only for educational purposes, providing a deep understanding of the neural networks mechanics without the complexity of advanced libraries. It features multiple basics activation functions, weight initializers, datasets, cost functions, optimizers, and utility functions.

## Features

- **Activation Functions:** LeakyReLU, ReLU, Sigmoid, Tanh, Linear, SoftMax
- **Initializers:** HeNormal, XavierNormal, Normal, Uniform
- **Datasets:** MNIST, FASHION MNIST, IRIS, MNIST BINARY
- **Cost Functions:** BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy
- **Optimizers:** SGD, Adam
- **Layers:** Dense
- **Models:** MLP
- **Utilities:** General utility functions for data manipulation and performance metrics


### When configuring your MLP model to use SoftMax activation function, follow steps below:

- **Last Layer Configuration:** Use a `Linear` activation function in the last dense layer. 
- **Loss Function:** Pair this configuration with the `CategoricalCrossEntropy` (CCE) loss. 
