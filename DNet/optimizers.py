import numpy as np


class SGD:
    """
    Implements the Stochastic gradient descent algorithm
    """
    def __init__(self, lr=0.0075):
        """
        Sets the learning rate.
        """
        self.lr = lr

    def optim(self, weigths, bias, dW, db):
        """
        Updates the weights and biases.
        """
        weigths = weigths - self.lr * dW
        bias = bias - self.lr * db

        return weigths, bias


class Adam:
    def __init__(self):
        pass