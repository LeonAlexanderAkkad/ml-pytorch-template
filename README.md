# Machine Learning PyTorch template
This template can be used to easily set up a ML project using PyTorch.

## Models
Various models can be found in the `models.py` file ranging from a simple FNN to a simple CNN, RNN and LSTM so far. A simple Transformer will be added to this file in the foreseeable future.

## Dataset
A simple pytorch Dataset class can be found in the `dataset.py` file including a potential transform chain in the `__getitem__` method.

## Training and Evaluation
A complete training and evaluation loop using tensorboard can be found in the `training.py` file. Learning rate scheduling is implemented manually, however it might be exchanged with the PyTorch scheduler. Moreover, accuracy measures and PR curves will be added to tensorboard and the computation of predictions might be transferred to a different file in the future.
