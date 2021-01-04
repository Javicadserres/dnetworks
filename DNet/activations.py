import numpy as np 


class ReLU:
    """
    Returns the values for a given activation function.
    """
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'ReLU'

    def forward(self, Z):
        """
        Computes the RELU function for a given Z.
        """
        self.relu = np.maximum(0, Z)

        return self.relu

    def backward(self, dA):
        """
        Backward propagation ReLU.
        """
        dZ = np.where(self.relu <= 0, 0, dA)

        return dZ


class Sigmoid:
    """
    Returns the values for a given activation function.
    """
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Sigmoid'

    def forward(self, Z):
        """
        Computes the RELU function for a given Z.
        """
        self.sigmoid = 1 / (1 + np.exp(- Z))

        return self.sigmoid

    def backward(self, dA):
        """
        Computes the backward propagation for Sigmoid.
        """
        dZ = dA * self.sigmoid * (1 - self.sigmoid)

        return dZ

class Tanh:
    def __init__(self):
        pass

class LeakyReLU:
    def __init__(self):
        pass