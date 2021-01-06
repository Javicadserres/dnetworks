import numpy as np 


class ReLU:
    """
    Returns the values for a given activation function.
    """
    def __init__(self):
        self.type = 'ReLU'

    def forward(self, Z):
        """
        Computes the RELU function for a given Z.
        """
        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dA):
        """
        Backward propagation ReLU.
        """
        dZ = dA * np.where(self.A <= 0, 0, 1)

        return dZ


class Sigmoid:
    """
    Returns the values for a given activation function.
    """
    def __init__(self):
        self.type = 'Sigmoid'

    def forward(self, Z):
        """
        Computes the RELU function for a given Z.
        """
        self.A = 1 / (1 + np.exp(- Z))

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation for Sigmoid.
        """
        dZ = dA * self.A * (1 - self.A)

        return dZ


class Tanh:
    """
    Returns the values for the hyperbolic tangent activation 
    function.
    """
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):
        """
        Computes the Tanh function for a given Z.
        """
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):
        """
        Backward propagation Tanh.
        """
        dZ = dA * (1 - np.power(self.A, 2))

        return dZ


class LeakyReLU:
    """
    Returns the values for a given activation function.
    """
    def __init__(self):
        self.type = 'LeakyReLU'

    def forward(self, Z):
        """
        Computes the Leaky ReLU function for a given Z.
        """
        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dA):
        """
        Backward propagation Leaky ReLU.
        """
        dZ = dA * np.where(self.A <= 0, 0.01, 1)

        return dZ


class Softmax:
    """
    Returns the values for a given activation function.
    """
    def __init__(self):
        self.type = 'Softmax'

    def forward(self, Z):
        """
        Computes the RELU function for a given Z.
        """
        t = np.exp(Z)
        self.A =  t / np.sum(t) 

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation for Sigmoid.
        """
        pass