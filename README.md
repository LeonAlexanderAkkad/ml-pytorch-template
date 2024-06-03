# Machine Learning PyTorch template
This template can be used to easily set up a ML project using PyTorch.

## Models
Various models can be found in the `models.py` file ranging from a simple FNN to a simple CNN, RNN and LSTM so far. A simple Transformer will be added to this file in the foreseeable future.

## Dataset
A simple pytorch Train and Test Dataset class can be found in the `dataset.py` file including a potential transform chain in the `__getitem__` method.

## Training and Evaluation
A complete training and evaluation loop can be found in the Trainer class found in`training.py`. Learning rate scheduling is implemented manually, however it might be exchanged with the PyTorch scheduler. Moreover, accuracy and loss will be logged to weights and biases.
