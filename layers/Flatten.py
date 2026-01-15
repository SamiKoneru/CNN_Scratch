import numpy as np

class Flatten:
    def __init__(self, B=0, H=0, W=0, C=0):
        self.B = B
        self.H = H
        self.W = W
        self.C = C
    
    def forward(self, input):
        self.B, self.H, self.W, self.C = input.shape
        return input.reshape(self.B, -1)

    def backward(self, dout):
        return dout.reshape(self.B, self.H, self.W, self.C)

    def __repr__(self):
        return 'Flatten'
