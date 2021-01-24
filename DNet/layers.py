import numpy as np


class LinearLayer:
    """
    Class that implements a Linear layer.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the parameters with the input_dimensions and the
        output dimensions.

        Parameters
        ----------
        input_dim : int
            Dimension of the inputs.
        output_dim : int
            Dimensions of the output.
        """
        self.weights, self.bias = self._initialize_parameters(
            input_dim, output_dim
        )
        self.type = 'Linear'  
        self.oparams = None  
        self.Z = None 
    
    def forward(self, A):
        """
        Computes the forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input from the x variables or the output of the 
            activations.
        """
        self.m = A.shape[-1]
        self.A = A.reshape(-1, self.m)
        
        self.Z = self.weights.dot(A) + self.bias

        return self.Z

    def backward(self, dZ):
        """
        Computes the backward propagation.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.
        """
        self.dW = (1 / self.m) * dZ.dot(self.A.T)
        self.db = np.mean(dZ, axis=1, keepdims=True)
        dA = self.weights.T.dot(dZ)
        
        return dA

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

    def _initialize_parameters(self, input_dim, output_dim):
        """
        Initialize parameters randomly. 
        
        Parameters
        ----------
        input_dim : int
            Dimension of the inputs.
        output_dim : int
            Dimensions of the output.

        Returns
        -------
        weights : numpy.array
        bias : numpy.array
        """
        np.random.seed(1)
        den = np.sqrt(input_dim)

        weights = np.random.randn(output_dim, input_dim) / den
        bias = np.zeros((output_dim, 1))

        return weights, bias