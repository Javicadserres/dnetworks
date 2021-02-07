import numpy as np
from .base import ConvBase


class MaxPooling2D(ConvBase):
    """
    Applies a 2D max pooling over an input signal composed of several
    input planes.

    This layer walks through the input matrix (with a given kernel 
    size) computing the maximum.

    Example
    -------
    >>> import numpy as np
    >>> from DNet.layers import MaxPooling2D

    >>> model = MaxPooling2D(kernel_size=(2, 2), stride=1, padding=0)
    >>> input = np.random.randn(3, 3, 1, 1)
    >>> output = model.forward(input)
    >>> output.shape
    (2, 2, 1, 1)
    """
    def __init__(
        self, 
        kernel_size, 
        stride, 
        padding,
        padding_constant=0,
    ):
        """
        Parameters
        ----------
        kernel_size : tuple
            Size of the convolving kernel.
        stride : int
            Stride of the convolution.
        padding : int, tuple
            Dimension of the pad added to both sides of the input. 
        padding_constant : int, default=0
            Number added to the pad. 

        Attributes
        -----------
        A_resize : numpy.array
            Input image A, resized. 
            As an example, if we have an A (input) of size 
            (10, 10, 1, 100) and a kernel of size (2, 2) then we
            would have a new matrix of size 
            (2 * 2, 100 * 1 * 10 * 10).
        """
        super(MaxPooling2D, self).__init__(
            kernel_size, stride, padding, padding_constant
        )
        self.channel = 1
        self.type = 'MaxPooling'  
    
    def forward(self, A):
        """
        Computes the convolutional forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input image. With dimensions (width, height, deep, N).

        Returns
        -------
        Z : numpy.array
            Convolution output.
        """
        super(MaxPooling2D, self).forward(A)

        self.A_reshape = self.A.reshape(
            self.m * self.in_channels, 1, self.in_height, self.in_width
        )
        self.A_resize = self._resize_image(self.A_reshape)

        self.out_resize = np.max(self.A_resize, axis=0)
        out = self.out_resize.reshape(
            self.out_height, self.out_width, self.m, self.in_channels
        )
        self.Z = out.transpose(2, 3, 0, 1)

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
        super(MaxPooling2D, self).backward(dZ)

        dZ_flat = self.dZ.transpose(2, 3, 0, 1).ravel()
        back_A_resize = (self.A_resize == dZ_flat) * dZ_flat

        dA = self._resize_matrix(back_A_resize)
        dA = dA.reshape(self.A.shape)

        return dA.T


class AveragePooling2D(ConvBase):
    """
    Applies a 2D max pooling over an input signal composed of several
    input planes.

    This layer walks through the input matrix (with a given kernel 
    size) computing the mean.

    Example
    -------
    >>> import numpy as np
    >>> from DNet.layers import AveragePooling2D

    >>> model = MaxPooling2D(kernel_size=(2, 2), stride=1, padding=0)
    >>> input = np.random.randn(3, 3, 1, 1)
    >>> output = model.forward(input)
    >>> output.shape
    (2, 2, 1, 1)
    """
    def __init__(
        self, 
        kernel_size, 
        stride, 
        padding,
        padding_constant=0,
    ):
        """
        Parameters
        ----------
        kernel_size : tuple
            Size of the convolving kernel.
        stride : int
            Stride of the convolution.
        padding : int, tuple
            Dimension of the pad added to both sides of the input. 
        padding_constant : int, default=0
            Number added to the pad. 

        Attributes
        -----------
        A_resize : numpy.array
            Input image A, resized. 
            As an example, if we have an A (input) of size 
            (10, 10, 1, 100) and a kernel of size (2, 2) then we
            would have a new matrix of size 
            (2 * 2, 100 * 1 * 10 * 10).
        """
        super(AveragePooling2D, self).__init__(
            kernel_size, stride, padding, padding_constant
        )
        self.channel = 1
        self.type = 'AveragePooling'  
    
    def forward(self, A):
        """
        Computes the convolutional forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input image. With dimensions (width, height, deep, N).

        Returns
        -------
        Z : numpy.array
            Convolution output.
        """
        super(AveragePooling2D, self).forward(A)

        self.A_reshape = self.A.reshape(
            self.m * self.in_channels, 1, self.in_height, self.in_width
        )
        self.A_resize = self._resize_image(self.A_reshape)

        self.out_resize = np.mean(self.A_resize, axis=0)
        out = self.out_resize.reshape(
            self.out_height, self.out_width, self.m, self.in_channels
        )
        self.Z = out.transpose(2, 3, 0, 1)

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
        super(AveragePooling2D, self).backward(dZ)

        dZ_flat = self.dZ.transpose(2, 3, 0, 1).ravel()
        equal_weight = dZ_flat / (self.k_height * self.k_width)

        back_A_resize = np.ones(self.A_resize.shape) * equal_weight

        dA = self._resize_matrix(back_A_resize)
        dA = dA.reshape(self.A.shape)

        return dA.T