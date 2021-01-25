import numpy as np


class Flatten:
    ''' Flatten layer used to reshape inputs into vector representation
    Layer should be used in the forward pass before a dense layer to
    transform a given tensor into a vector.
    '''

    def __init__(self):
        self.type = 'Flatten'

    def forward(self, A):
        ''' Reshapes a n-dim representation into a vector
            by preserving the number of input rows.
        Examples:
            [10000,[1,28,28]] -> [10000,784]
        '''
        self.A = A
        self.Z = A.reshape(-1, self.A.shape[-1])

        return self.Z

    def backward(self, dA):
        ''' Restore dimensions before flattening operation
        '''
        dZ = dA.reshape(self.A.shape)
        return dZ