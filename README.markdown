# CNN Scratch Notebook

This folder contains a collection of files composing the structure of a simple convolutional neural network implementation from first principles. The goal is to step through the building blocks (convolutions, activations, simple training loop) without relying on high-level deep learning frameworks and libraries.

## Prerequisites
- Python 3.9+ recommended
- Jupyter Notebook or JupyterLab
- Common scientific stack (see the imports inside the notebook, e.g., `numpy`, `matplotlib`)

## Project structure
- `layers` — folder for all neural network layer classes (convolution, pooling, dense, flatten).
- `activations` - folder for activation functions (ReLU, softmax).
- `Network.py` - class for creating, structuring, training, and testing the network.
- `mnist.npz` - data file.
- `README.markdown` — this overview.
