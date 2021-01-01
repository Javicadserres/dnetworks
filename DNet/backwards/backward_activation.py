import numpy as np
from activations import sigmoid


def relu_backward(grad_X, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Parameters
    ----------
    Z : np.array
        Where we store for computing backward propagation 
        efficiently.

    Returns
    -------
    grad_Z : np.array
        Gradient of the cost with respect to Z.
    """
    grad_Z = np.where(Z < 0, 0, grad_X)
    
    return grad_Z


def sigmoid_backward(grad_X, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Parameters
    ----------
    grad_X : np.array
        Post-activation gradient.
    Z : np.array
        Z values stored for computing the backward propagation.

    Returns
    -------
    grad_Z -- Gradient of the cost with respect to Z
    """    
    sig = sigmoid(Z)
    grad_Z = grad_X * sig * (1 - sig)
    
    return grad_Z
