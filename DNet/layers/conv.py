import numpy as np
from .base import ConvBase


class Conv2D(ConvBase):
    """
    Applies a 2D convolution over an input signal composed of several
    input planes.

    The formula of the convolution can be described as:

    .. math::
        \text{out}(C_{\text{out}_j}, N_i) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(k, C_{\text{out}_j}) \star \text{input}(k, N_i)

    where :math:`\star` is the valid 2D `cross-correlation` operator,
    :math:`N` is a batch size and :math:`C` denotes a number of 
    channels. 

    References
    ----------
    [1] PyTorch - CONV2D:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    Examples
    --------
    >>> import numpy as np
    >>> from DNet.layers import Conv2D

    >>> model = Conv2D(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(2, 2), 
            stride=1, 
            padding=0
        )
    >>> input = np.random.randn(3, 3, 1, 1)
    >>> output = model.forward(input)
    >>> output.shape
    (2, 2, 1, 1)
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
        kernel_size : tuple
            Size of the convolving kernel.
        stride : int
            Stride of the convolution.
        padding : int, tuple
            Dimension of the pad added to both sides of the input. 
        padding_constant : int, default=0
            Number added to the pad. 

        Atributes
        ---------
        weights : numpy.array
            Weight parameters of the layer.
        bias : numpy.array
            Bias parameters of the layer.
        A_resize : numpy.array
            Input image A, resized.
            As an example, if we have an A (input) of size 
            (10, 10, 1, 100) and a kernel of size (2, 2) then we
            would have a new matrix of size 
            (2 * 2, 100 * 1 * 10 * 10).
        """
        super(Conv2D, self).__init__(
            kernel_size, stride, padding, padding_constant
        )
        self.weights, self.bias = self._initialize_parameters(
                in_channels, out_channels
        )
        self.in_channels = in_channels
        self.channel = in_channels
        self.out_channels = out_channels
        self.oparams = None
        self.type = 'Convolutional'  
    
    def forward(self, A):
        """
        Computes the convolutional forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input image. With dimensions (W, H, channels, N).

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
            The gradient of the input image with respect to the 
            current layer.
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