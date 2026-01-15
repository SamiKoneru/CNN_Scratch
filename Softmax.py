import numpy as np


class Softmax:
    def forward(self, x):
        max_x = np.max(x, axis=1, keepdims=True)
        shifted = x - max_x
        x_exp = np.exp(shifted)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def backward(self, dout):
        return dout

    def __repr__(self):
        return "Softmax"