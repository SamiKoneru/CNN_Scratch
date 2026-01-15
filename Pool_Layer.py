import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Pool_Layer:
    def __init__(self, H=2, W=2, stride=2):
        self.H = H
        self.W = W
        self.stride = stride
        self.maxes = 0
        self.x_shape = 0

    def forward(self, x):
        patches = sliding_window_view(x, (self.H, self.W), axis=(1, 2))[:, ::self.stride, ::self.stride]  # shape: (B, H_out, W_out, C_in, self.H, self.W)
        patches = np.moveaxis(patches, 3, -1)

        B, out_H, out_W, _, _, C = patches.shape
        flat_patches = np.reshape(patches, (B, out_H, out_W, self.H * self.W, C))
        
        self.x_shape = x.shape
        self.maxes = np.argmax(flat_patches, axis=3)

        out = np.take_along_axis(flat_patches, self.maxes[:, :, :, None, :], axis=3)
        out = np.squeeze(out, axis=3)

        return out

    def backward(self, dout):
        B, out_H, out_W, C = self.maxes.shape
        h_max, w_max = np.unravel_index(self.maxes, (self.H, self.W))

        h_indices = np.arange(out_H)[None, :, None, None] * self.stride + h_max
        w_indices = np.arange(out_W)[None, None, :, None] * self.stride + w_max
        b_indices = np.arange(B)[:, None, None, None]
        c_indices = np.arange(C)[None, None, None, :]


        dX = np.zeros(self.x_shape)
        np.add.at(dX, (b_indices, h_indices, w_indices, c_indices), dout)

        return dX

    def __repr__(self):
        return f"Pooling: {self.H}x{self.W}, stride={self.stride}"