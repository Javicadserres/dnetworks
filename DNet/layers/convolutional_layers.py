import numpy as np
from layers.layer import ConvBase


class Conv2D(ConvBase):
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
        super(Conv2D, self).__init__(
            kernel_size, stride, padding, padding_constant
        )
        self.weights, self.bias = self._initialize_parameters(
                in_channels, out_channels
        )
        self.oparams = None
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        super(Conv2D, self).forward(A)

        self.A_resize = self._resize_image(self.A)
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
        super(Conv2D, self).backward(dZ)
        # bias
        self.db = np.sum(self.dZ, axis=(0, 2, 3))
        self.db = self.db.reshape(self.out_channels, -1)

        dZ_resize = self.dZ.transpose(1, 2, 3, 0)
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

    def _initialize_parameters(self, in_channels, out_channels):
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