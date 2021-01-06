import numpy as np
from utils import initialize_parameters


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
        self.weights, self.bias = initialize_parameters(
            input_dim, output_dim
        )
        self.type = 'Linear'  
        self.oparams = None   
    
    def forward(self, A):
        """
        Computes the forward propagation.
        
        Parameters
        ----------
        A : numpy.array
            Input from the x variables or the output of the 
            activations.
        """
        self.A = A
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
        m = self.A.shape[1]

        self.dW = (1 / m) * dZ.dot(self.A.T)
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