import numpy as np
from padding_layers import ConstantPad


class MaxPooling2D:
    """
    Applies a 2D max pooling over an input signal composed of several
    input planes.
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
        self.type = 'MaxPooling'  
    
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
        self.A = A
        (
            self.m, 
            self.in_channels, 
            self.in_height, 
            self.in_width
        ) = A.shape

        self.A_reshape = A.reshape(
            self.m * self.in_channels, 1, self.in_height, self.in_width
        )
        self.A_resize = self._resize_image(self.A_reshape)

        self.out_resize = np.max(self.A_resize, axis=0)
        out = self.out_resize.reshape(
            self.out_height, self.out_width, self.m, self.in_channels
        )
        self.Z = out.transpose(2, 3, 0, 1)

        return self.Z

    def backward(self, dA):
        """
        Computes the backward propagation of the convolutional layer.

        Parameters
        ----------
        dA : numpy.array
            The gradient of the of the output with respect to the
            next layer.
        
        Returns
        -------
        dZ : numpy.array
            The gradient of the convolutional layer.
        """
        dA_flat = dA.transpose(2, 3, 0, 1).ravel()
        back_A_resize = (self.A_resize == dA_flat) * dA_flat

        dZ = self._resize_matrix(back_A_resize)
        dZ = dZ.reshape(self.A.shape)

        return dZ

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
        channels_i, height_i, width_i = self._index_resize(A)
        A_resize = A_padded[:, channels_i, height_i, width_i]

        new_shape = self.k_height * self.k_width
        A_resize = A_resize.transpose(1, 2, 0).reshape(new_shape, -1)

        return A_resize

    def _index_resize(self, A):
        """
        Select the index to reshape the image into a 2D matrix.

        Returns
        -------
        channels_i : numpy.array
        height_i : numpy.array
        width_i : numpy.array
        """ 
        _, in_channels, _, _ = A.shape
        self.out_height, self.out_width = self._get_size_out()

        # height
        height_i0 = np.repeat(np.arange(self.k_height), self.k_width)
        height_i0 = np.tile(height_i0, in_channels)

        height_i1 = np.repeat(
            np.arange(self.out_height), self.out_width
        )
        height_i1 *= self.stride
        self.height_i = height_i0.reshape(-1, 1) + height_i1

        # width 
        height_channel = self.k_height * in_channels
        width_i0 = np.tile(np.arange(self.k_width), height_channel)

        width_i1 = np.tile(
            np.arange(self.out_width), self.out_height
        )
        width_i1 *= self.stride
        self.width_i = width_i0.reshape(-1, 1) + width_i1 

        # channels
        k_square = self.k_height * self.k_width
        channels_i = np.repeat(np.arange(in_channels), k_square)
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

        m_channel = self.m * self.in_channels
        A_padded = np.zeros((m_channel, 1, H_padded, W_padded))

        shape_len = self.k_height * self.k_width
        dA_resize = dA_resize.reshape(shape_len, -1, m_channel)
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


class AveragePooling2D:
    """
    Applies a 2D max pooling over an input signal composed of several
    input planes.
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
        self.type = 'AveragePooling'  
    
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
        self.A = A
        (
            self.m, 
            self.in_channels, 
            self.in_height, 
            self.in_width
        ) = A.shape

        self.A_reshape = A.reshape(
            self.m * self.in_channels, 1, self.in_height, self.in_width
        )
        self.A_resize = self._resize_image(self.A_reshape)

        self.out_resize = np.mean(self.A_resize, axis=0)
        out = self.out_resize.reshape(
            self.out_height, self.out_width, self.m, self.in_channels
        )
        self.Z = out.transpose(2, 3, 0, 1)

        return self.Z

    def backward(self, dA):
        """
        Computes the backward propagation of the convolutional layer.

        Parameters
        ----------
        dA : numpy.array
            The gradient of the of the output with respect to the
            next layer.
        
        Returns
        -------
        dZ : numpy.array
            The gradient of the convolutional layer.
        """
        dA_flat = dA.transpose(2, 3, 0, 1).ravel()
        equal_weight = dA_flat / (self.k_height * self.k_width)

        back_A_resize = np.ones(self.A_resize.shape) * equal_weight

        dZ = self._resize_matrix(back_A_resize)
        dZ = dZ.reshape(self.A.shape)

        return dZ

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
        channels_i, height_i, width_i = self._index_resize(A)
        A_resize = A_padded[:, channels_i, height_i, width_i]

        new_shape = self.k_height * self.k_width
        A_resize = A_resize.transpose(1, 2, 0).reshape(new_shape, -1)

        return A_resize

    def _index_resize(self, A):
        """
        Select the index to reshape the image into a 2D matrix.

        Returns
        -------
        channels_i : numpy.array
        height_i : numpy.array
        width_i : numpy.array
        """ 
        _, in_channels, _, _ = A.shape
        self.out_height, self.out_width = self._get_size_out()

        # height
        height_i0 = np.repeat(np.arange(self.k_height), self.k_width)
        height_i0 = np.tile(height_i0, in_channels)

        height_i1 = np.repeat(
            np.arange(self.out_height), self.out_width
        )
        height_i1 *= self.stride
        self.height_i = height_i0.reshape(-1, 1) + height_i1

        # width 
        height_channel = self.k_height * in_channels
        width_i0 = np.tile(np.arange(self.k_width), height_channel)

        width_i1 = np.tile(
            np.arange(self.out_width), self.out_height
        )
        width_i1 *= self.stride
        self.width_i = width_i0.reshape(-1, 1) + width_i1 

        # channels
        k_square = self.k_height * self.k_width
        channels_i = np.repeat(np.arange(in_channels), k_square)
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

        m_channel = self.m * self.in_channels
        A_padded = np.zeros((m_channel, 1, H_padded, W_padded))

        shape_len = self.k_height * self.k_width
        dA_resize = dA_resize.reshape(shape_len, -1, m_channel)
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


