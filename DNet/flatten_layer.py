import numpy as np


class Flatten():
    ''' Flatten layer used to reshape inputs into vector representation
    Layer should be used in the forward pass before a dense layer to
    transform a given tensor into a vector.
    '''

    def __init__(self):
        self.type = 'Flatten'

    def forward(self, X):
        ''' Reshapes a n-dim representation into a vector
            by preserving the number of input rows.
        Examples:
            [10000,[1,28,28]] -> [10000,784]
        '''
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.reshape(-1).reshape(self.out_shape)
        return out

    def backward(self, dout):
        ''' Restore dimensions before flattening operation
        '''
        out = dout.reshape(self.X_shape)
        return out, []