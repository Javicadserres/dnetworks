import numpy as np 
from .base import Base


class ReLU(Base):
    """
    Class for the ReLU activation function.

    Applies the maximum between the given input and a 0:

    :math:`ReLU(x) = max(0, x)`    
    """
    def __init__(self):
        self.type = 'ReLU'

    def forward(self, Z):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Z : numpy.array
            Input.

        Returns
        -------
        A : numpy.array
            Output.
        """
        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.

        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.

        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        dZ = dA * np.where(self.A <= 0, 0, 1)

        return dZ


class Sigmoid(Base):
    """
    Class for the Sigmoid activation function.

    Applies the sigmoid function:

    :math:`Sigmoid(x) = \frac{1}{1 + e^{-x}}`    
    """
    def __init__(self):
        self.type = 'Sigmoid'

    def forward(self, Z):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Z : numpy.array
            Input.

        Returns
        -------
        A : numpy.array
            Output.
        """
        self.A = 1 / (1 + np.exp(- Z))

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.

        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.

        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        dZ = dA * self.A * (1 - self.A)

        return dZ


class Tanh(Base):
    """
    Class for the Hyperbolic tangent activation function.

    Applies the hyperbolic tangent function:

    :math:`Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}`  
    """
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Z : numpy.array
            Input.

        Returns
        -------
        A : numpy.array
            Output.
        """
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.

        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.

        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        dZ = dA * (1 - np.power(self.A, 2))

        return dZ


class LeakyReLU(Base):
    """
    Class for the LeakyReLU activation function.

    Applies the maximum between the given input and a 0:

    :math:`LeakyReLU(x) = max(slope * x, x)`
    """
    def __init__(self, slope=0.01):
        """
        Parameters
        ----------
        slope : int
            Slope to use for the leaky relu.
        """
        self.type = 'LeakyReLU'
        self.slope = slope

    def forward(self, Z):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Z : numpy.array
            Input.

        Returns
        -------
        A : numpy.array
            Output.
        """
        self.A = np.maximum(Z, self.slope * Z)

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.

        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.

        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        dZ = dA * np.where(self.A <= 0, self.slope, 1)

        return dZ


class Softmax(Base):
    """
    Class for the Softmax activation function.

    Applies the softmax function to a given input:

    :math:`Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}e^{x_j}}`
    """
    def __init__(self):
        self.type = 'Softmax'
        self.eps = 1e-15

    def forward(self, Z):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Z : numpy.array
            Input.

        Returns
        -------
        A : numpy.array
            Output.
        """
        self.Z = Z

        t = np.exp(Z - np.max(Z, axis=0))
        self.A =  t / np.sum(t, axis=0, keepdims=True)

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.

        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.

        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        n, m = self.A.shape

        matrix1 = np.einsum('ji,ki->jki', self.A, self.A) 
        matrix2 = np.einsum('ji,jk->jki', self.A, np.eye(n, n))

        dSoftmax = matrix2 - matrix1
        dZ = np.einsum('jki,ki->ji', dSoftmax, dA)
        
        return dZ