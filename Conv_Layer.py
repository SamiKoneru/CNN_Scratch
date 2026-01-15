import numpy as np
from ReLU import ReLU
from numpy.lib.stride_tricks import sliding_window_view


def convolve(x, kernel, bias, stride, padding):
    if padding > 0:
        x = np.pad(x, pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    B, H, W, C_in = x.shape
    KH, KW, _, C_out = kernel.shape

    patches = sliding_window_view(x, (KH, KW), axis=(1, 2))[:, ::stride, ::stride]  # shape: (B, H_out, W_out, C_in, KH, KW)
    patches = np.moveaxis(patches, 3, -1)

    _, H_out, W_out, _, _, _ = patches.shape
    patches = np.reshape(patches, (B, H_out, W_out, KH*KW*C_in))

    flat_kernel = np.reshape(kernel, (KH*KW*C_in, C_out))


    out = patches @ flat_kernel

    if bias is not None:
        out += bias

    return out, patches

class Conv_Layer:
    def __init__(self, C_in=1, C_out=1, activation='relu', H=3, W=3, stride=1, padding=0, lrate=0.01):
        self.kernel = np.random.normal(0, np.sqrt(2/(H*W*C_in)), (H, W, C_in, C_out))
        self.bias = np.zeros((C_out))
        self.stride = stride
        self.padding = padding
        self.lrate = lrate
        self.last_in = 0
        self.patches = 0
        if activation == 'relu':
            self.activation = ReLU()
        else:
            self.activation = None
        

    def forward(self, x):
        self.last_in = x
        out, self.patches = convolve(x, self.kernel, self.bias, self.stride, self.padding)
        if self.activation:
            return self.activation.forward(out)
        else:
            return out


    def backward(self, dout):    # need to review concepts
        if self.activation:
            dout = self.activation.backward(dout)
        
        K, _, _, _ = self.kernel.shape
        B, _, _, _ = self.last_in.shape
        _, dout_H, dout_W, dout_C = dout.shape

        db = dout.sum(axis=(0,1, 2)) / B
        self.bias = self.bias - db * self.lrate


        flat_patches = np.reshape(self.patches, (B * dout_H * dout_W, -1))
        flat_dout = np.reshape(dout, (B * dout_H * dout_W, -1))

        flat_dW = flat_patches.T @ flat_dout
        dW = np.reshape(flat_dW, self.kernel.shape) / B

        self.kernel = self.kernel - dW * self.lrate
        

        if self.stride > 1:
            dilated_dout = np.zeros((B, (dout_H - 1) * self.stride + 1, (dout_W - 1) * self.stride + 1, dout_C))
            dilated_dout[:, ::self.stride, ::self.stride, :] = dout
        else:
            dilated_dout = dout

        backpad = K-1-self.padding
        dX, _ = convolve(dilated_dout, np.flip(self.kernel, axis=(0, 1)).swapaxes(2, 3), None, self.stride, backpad)
        
        return dX

    def __repr__(self):
        return f"Convolution: shape={self.kernel.shape}, activation={str(self.activation)}, stride={self.stride}, padding={self.padding}"