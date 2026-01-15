import numpy as np

class ReLU:
    def __init__(self):
        self.last_in = 0

    def forward(self, x):
        self.last_in = x
        return np.maximum(0, x)

    def backward(self, dout):
        return np.where(self.last_in<=0, 0, dout)

    def __repr__(self):
        return "ReLU"