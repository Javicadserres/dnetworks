import numpy as np


class ConstantPad:
    """
    Pads the input tensor boundaries with a constant value.

    Example
    -------
    >>> import numpy as np
    >>> from dnetworks.layers import ConstantPad
    
    >>> input = np.ones(shape=(2, 2))
    >>> model = ConstantPad(padding=1, dim=2, constant=0)
    >>> output = model.pad(input)
    >>> output.shape
    (4, 4)
    """
    def __init__(self, padding, dim, constant):
        """
        Parameters
        ----------
        padding : int, tuple
            The size of the padding. If is int, uses the same padding
            in all boundaries.
        dim : int
            Dimension of the pad. 
        constant : int, defualt=0
            The constant to add to the boundries.
        """
        self.padding = self._int2tuple(padding)
        self.dim = dim
        self.constant = constant

    def pad(self, X):
        """
        Returns input padded.

        Parameters
        ----------
        X : numpy.array
            Input
        """
        self.tuples = self._get_tuples(X, self.padding, self.dim)

        padded = np.pad(
            X, 
            self.tuples, 
            mode='constant', 
            constant_values=self.constant
        ) 

        return padded
   
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
        X : numpy.array
            Input.
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