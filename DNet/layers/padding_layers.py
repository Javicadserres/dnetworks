import numpy as np


class ConstantPad:
    """
    Pads the input tensor boundaries with a constant value.
    """
    def __init__(self, X, padding, dim, constant):
        """
        Parameters
        ----------
        X : np.array
            Input
        padding : int, tuple
            The size of the padding. If is int, uses the same padding
            in all boundaries.
        dim : int
            Dimension of the pad. 
        constant : int, defualt=0
            The constant to add to the boundries.
        """
        padding = self._int2tuple(padding)
        tuples = self._get_tuples(X, padding, dim)

        self.padded = np.pad(
            X, tuples, mode='constant', constant_values=constant
        ) 
    def pad(self):
        """
        Returns input padded.
        """
        return self.padded
   
    def _int2tuple(self, padding):
        """
        Converts the integer into tuple.

        Parameters
        ----------
        padding : int, tuple
            The size of the padding. If is int, uses the same padding in 
            all boundaries.
        
        Returns
        -------
        padding : tuple
        """
        if type(padding) is int: padding = (padding, padding)

        return padding
    
    def _get_tuples(self, X, padding, dim):
        """
        Get the tuples to introduce for the pad parameters.

        Parameters
        ----------
        X : np.array
            Input
        padding : int, tuple
            The size of the padding. If is int, uses the same padding in 
            all boundaries.
        dim : int
            Dimension of the pad. 

        Returns
        -------
        tuples : tuple
        """
        len_tuples = len(X.shape) - dim
        tuples = ((0, 0), ) * len_tuples
        tuples += (padding, padding)

        return tuples
