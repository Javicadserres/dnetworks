import numpy as np
from utils import initialize_parameters


class LinearLayer:
    """
    Linear layer.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the parameters with the input_dimensions and the
        output dimensions.
        """
        self.weights, self.bias = initialize_parameters(
            input_dim, output_dim
        )
        self.type = 'Linear'        
    
    def forward(self, A):
        """
        Implements the forward propagation of a linear layer.
        """
        self.A = A
        self.Z = self.weights.dot(A) + self.bias

        return self.Z

    def backward(self, dZ):
        m = self.A.shape[1]

        self.dW = (1 / m) * dZ.dot(self.A.T)
        self.db = np.mean(dZ, axis=1, keepdims=True)
        dA = self.weights.T.dot(dZ)
        
        return dA

    def optimize(self, method='SGD', lr=0.0075):
        if method=='SGD':
            self.weights = self.weights - lr * self.dW
            self.bias = self.bias - lr * self.db