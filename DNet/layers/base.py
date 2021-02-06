import numpy as np
from .padding import ConstantPad

class Base(object):
    """
    Main class for all the neural network layers. The models should
    also subclass this class.

    Example
    -------
    >>> import numpy as np
    >>> from DNet.layers import Base

    >>> class TanhModel(Base):
         def __init__(self, input_dim, output_dim):
            self.input_dim = input_dim
            self.output_dim = output_dim
        def forward(self, X):
            A = np.tanh(X)
            return A
    """
    def __init__(self):
        raise NotImplementedError()

    def forward(self):
        """
        Computes the forward propagation of the layer.
        """
        raise NotImplementedError()

    def backward(self):
        """
        Computes the backward propagation of the layer. 
        """
        raise NotImplementedError()


class ConvBase(Base):
    """
    Main class for convolutional layers.
    """
    def __init__(
        self, 
        kernel_size, 
        stride, 
        padding,
        padding_constant
    ):
        """
        Parameters
        ----------
        kernel_size : int, tuple
            Size of the convolving kernel.
        stride : int
            Stride of the convolution.
        padding : int, tuple
            Dimension of the pad added to both sides of the input. 
        padding_constant : int, default=0
            Number added to the pad. 
        """
        self.k_height, self.k_width = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_constant = padding_constant

    def forward(self, A):
        """
        Computes the convolutional forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input image.
        """
        self.A = A.T
        (
            self.m, 
            self.in_channels, 
            self.in_height, 
            self.in_width
        ) = self.A.shape

        pass
        
    def backward(self, dZ):
        """
        Computes the backward propagation of the convolutional layer.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.
        """
        self.dZ = dZ.T

        pass

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
            self.padding, dim=2, constant=self.padding_constant
        )
        A_padded = constpad.pad(A)
        # resize image
        channels_i, height_i, width_i = self._index_resize()
        A_resize = A_padded[:, channels_i, height_i, width_i]

        new_shape = self.k_height * self.k_width * self.channel
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
        height_i0 = np.tile(height_i0, self.channel)

        height_i1 = np.repeat(
            np.arange(self.out_height), self.out_width
        )
        height_i1 *= self.stride
        self.height_i = height_i0.reshape(-1, 1) + height_i1

        # width 
        height_channel = self.k_height * self.channel
        width_i0 = np.tile(np.arange(self.k_width), height_channel)

        width_i1 = np.tile(
            np.arange(self.out_width), self.out_height
        )
        width_i1 *= self.stride
        self.width_i = width_i0.reshape(-1, 1) + width_i1 

        # channels
        k_square = self.k_height * self.k_width
        channels_i = np.repeat(np.arange(self.channel), k_square)
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
        
        m_channel = self.m * self.in_channels / self.channel
        m_channel = int(m_channel)
 
        A_padded = np.zeros(
            (m_channel, self.channel, H_padded, W_padded)
        )

        shape_len = self.k_height * self.k_width * self.channel
        dA_resize = dA_resize.reshape(shape_len, -1, m_channel)
        dA_resize = dA_resize.transpose(2, 0, 1)

        indices = (
            slice(None), self.channels_i, self.height_i, self.width_i
        )
        np.add.at(A_padded, indices , dA_resize)

        if self.padding == 0:
            return A_padded

        return A_padded[
            :, :, self.padding:-self.padding, self.padding:-self.padding
            ]