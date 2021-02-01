import numpy as np
from .base import Base


class Flatten(Base):
    """
    Flattens a contiguous range of dims into a vector.
    """
    def __init__(self):
        self.type = 'Flatten'

    def forward(self, A):
        """
        Reshapes a n-dim representation into a vector, keeping all
        the values.

        Parameters
        ----------
        A : numpy.array
            Array that will be reshaped.

        Returns
        -------
        Z : numpy.array
            Reshaped array
        
        Example
        -------
        >> from flatten_layer import Flatten

        >> dim_array = np.zeros((2, 2, 1, 4))
        >> flat = Flatten()
        >> vector = flat.forward(dim_array)
        """
        self.A = A
        self.Z = A.reshape(-1, self.A.shape[-1])

        return self.Z

    def backward(self, dA):
        """
        Computes the backward propagation for the flatten layer.

        Parameters
        ----------
        dA : numpy.array
            Array containing the gradientes to be backpropagated.


        Returns
        -------
        dZ : numpy.array
            Array containing the  new gradients.
        """
        dZ = dA.reshape(self.A.shape)
        
        return dZ