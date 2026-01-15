import numpy as np
from ReLU import ReLU
from Softmax import Softmax

class Dense_Layer:
    def __init__(self, x_in=0, y_out=0, activation='relu', lrate=0.01):
        self.B = np.zeros((y_out))
        self.lrate = lrate
        self.last_in = 0
        if activation == 'relu':
            self.W = np.random.normal(0, np.sqrt(2.0 / x_in), (y_out, x_in))
            self.activation = ReLU()
        elif activation == 'softmax':
            limit = np.sqrt(6.0 / (x_in + y_out))
            self.W = np.random.uniform(-limit, limit, (y_out, x_in))
            self.activation = Softmax()
        else:
            limit = np.sqrt(6.0 / (x_in + y_out))
            self.W = np.random.uniform(-limit, limit, (y_out, x_in))
            self.activation = None

    def forward(self, x):
        self.last_in = x
        out = x @ self.W.T + self.B
        if self.activation:
            out = self.activation.forward(out)
        return out

    def backward(self, dout):
        B, _ = self.last_in.shape
        if self.activation:
            dout = self.activation.backward(dout)
        dW = (dout.T @ self.last_in) / B
        dX = dout @ self.W
        dB = dout.sum(axis=0) / B

        self.W = self.W - dW * self.lrate
        self.B = self.B - dB * self.lrate
        return dX

    def __repr__(self):
        return f"Dense: shape={self.W.shape}, activation={str(self.activation)}, lrate={self.lrate}"
