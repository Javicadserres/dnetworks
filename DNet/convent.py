import numpy as np
from padding_layers import ConstantPad


class Conv2D:
    """
    Applies a 2D convolution over an input signal composed of several
    input planes.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding,
        padding_constant=0
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int, tuple
            Size of the convolving kernel.
        stride : int
            Stride of the convolution.
        padding : int, tuple
            Dimension of the pad added to both sides of the input. 
        padding_constant : int, default=0
            Number added to the pad. 
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_height, self.k_width = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_constant = padding_constant
        self.weights, self.bias = self._initalize_parameters(
            in_channels, out_channels
        )
        self.oparams = None
        self.type = 'Convolutional'  
    
    def forward(self, A):
        """
        Computes the convolutional forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input image.

        Returns
        -------
        Z : numpy.array
            Convolution output.
        """
        A = A.T
        self.m, _, self.in_height, self.in_width = A.shape

        self.A_resize = self._resize_image(A)
        weghts_resize = self.weights.reshape(self.out_channels, -1)

        out = np.add(np.dot(weghts_resize, self.A_resize), self.bias)
        out = out.reshape(
            self.out_channels, self.out_height, self.out_width, self.m
        )
        self.Z = out.transpose(3, 0, 1, 2)

        return self.Z.T

    def backward(self, dZ):
        """
        Computes the backward propagation of the convolutional layer.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.
        
        Returns
        -------
        dA : numpy.array
            The gradient of the convolutional layer.
        """
        dZ = dZ.T
        # bias
        self.db = np.sum(dZ, axis=(0, 2, 3))
        self.db = self.db.reshape(self.out_channels, -1)

        dZ_resize = dZ.transpose(1, 2, 3, 0)
        dZ_resize = dZ_resize.reshape(self.out_channels, -1)

        # weights
        self.dW = np.dot(dZ_resize, self.A_resize.T)
        self.dW = self.dW.reshape(self.weights.shape)
        weight_resize = self.weights.reshape(self.out_channels, -1)

        # gradient
        dA_resize = np.dot(weight_resize.T, dZ_resize)
        dA = self._resize_matrix(dA_resize)

        return dA.T

    def optimize(self, method):
        """
        Updates the parameters with a given optimization method.

        Parameters
        ----------
        method : DNet.optimizers
            Optimization method to use to update the parameters.
        """
        self.weights, self.bias, self.oparams = method.optim(
            self.weights, self.bias, self.dW, self.db, self.oparams
        )

    def _initalize_parameters(self, in_channels, out_channels):
        """
        Initialize parameters of the convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution
        
        Returns
        -------
        weights : numpy.array
        bias : numpy.array
        """
        np.random.seed(1)

        # weights
        den = np.sqrt(in_channels * self.k_height * self.k_width)
        weights = np.random.randn(
            out_channels, in_channels, self.k_height, self.k_width
        ) / den

        # bias
        bias = np.zeros((out_channels, 1))

        return weights, bias

    def _get_size_out(self):
        """
        Get the new size of the ouput given the kernel size, padding,
        and stride parameters of the convolutional layer.

        Returns
        -------
        out_height :  int
        out_width : int
        """
        den = (2 * self.padding + self.stride) / self.stride

        # height
        diff_height = self.in_height - self.k_height
        out_height = diff_height / self.stride + den

        # width
        diff_width = self.in_width - self.k_width
        out_width = diff_width / self.stride + den

        return int(out_height), int(out_width)
        
    def _resize_image(self, A):
        """
        Transforms the image with dimensions, m, number of channels,
        image height and image width into a 2D dimension like in 
        order to compute faster the forward propagation.

        As an example, if we have an A (input) of size 
        (100, 1, 10, 10) and a kernel of size (2, 2) then we would
        have a new matrix of size (2 * 2, 100 * 1 * 10 * 10)

        Parameters
        ----------
        A : numpy array
            Image.
        
        Returns
        -------
        A_resize : numpy array
            Image resized.
        """
        # pad image
        constpad = ConstantPad(
            A, self.padding, dim=2, constant=self.padding_constant
        )
        A_padded = constpad.pad()
        # resize image
        channels_i, height_i, width_i = self._index_resize()
        new_shape = self.k_height * self.k_width * self.in_channels

        A_resize = A_padded[:, channels_i, height_i, width_i]
        A_resize = A_resize.transpose(1, 2, 0).reshape(new_shape, -1)

        return A_resize

    def _index_resize(self):
        """
        Select the index to reshape the image into a 2D matrix.

        Returns
        -------
        channels_i : numpy.array
        height_i : numpy.array
        width_i : numpy.array
        """ 
        self.out_height, self.out_width = self._get_size_out()

        # height
        height_i0 = np.repeat(np.arange(self.k_height), self.k_width)
        height_i0 = np.tile(height_i0, self.in_channels)

        height_i1 = np.repeat(
            np.arange(self.out_height), self.out_width
        )
        height_i1 *= self.stride
        self.height_i = height_i0.reshape(-1, 1) + height_i1

        # width 
        height_channel = self.k_height * self.in_channels
        width_i0 = np.tile(np.arange(self.k_width), height_channel)

        width_i1 = np.tile(
            np.arange(self.out_width), self.out_height
        )
        width_i1 *= self.stride
        self.width_i = width_i0.reshape(-1, 1) + width_i1 

        # channels
        k_square = self.k_height * self.k_width
        channels_i = np.repeat(np.arange(self.in_channels), k_square)
        self.channels_i = channels_i.reshape(-1, 1)

        return self.channels_i, self.height_i, self.width_i

    def _resize_matrix(self, dA_resize):
        """
        Transforms the 2D matrix size back to the image size.

        Parameters
        ----------
        dA_resize : numpy.array

        Returns
        -------
        Matrix with the image like size.
        """
        H_padded = self.in_height + 2 * self.padding
        W_padded = self.in_width + 2 * self.padding

        new_shape = (self.m, self.in_channels, H_padded, W_padded)
        A_padded = np.zeros(new_shape)

        shape_len = self.in_channels * self.k_height * self.k_width
        dA_resize = dA_resize.reshape(shape_len, -1, self.m)
        dA_resize = dA_resize.transpose(2, 0, 1)

        indices = (
            slice(None), self.channels_i, self.height_i, self.width_i
        )
        np.add.at(A_padded, indices , dA_resize)

        if self.padding == 0:
            return A_padded

        return A_padded[
            :, 
            :, 
            self.padding:-self.padding, 
            self.padding:-self.padding
        ]