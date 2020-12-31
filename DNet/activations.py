import numpy as np 


def sigmoid(Z):
    """
    Computes the sigmoid for a given Z.

    Parameters
    ----------
    Z : np.array

    Returns
    -------
    sig : np.array
        The sigmoid for the given Z values.
    """
    sig = 1 / (1 + np.exp(-Z))

    return sig


def tanh(Z):
    """
    Computes de hyperbolic tangent for a given Z.

    Parameters
    ----------
    Z : np.array

    Returns
    -------
    tan : np.array
        The hyperbolic tangent for the given Z.
    """
    tan = np.tanh(Z)

    return tan


def relu(Z):
    """
    Computes the RELU function for a given Z.

    Parameters
    ----------
    Z : np.array

    Returns
    -------
    rel : np.array
    """
    rel = np.maximum(0, Z)

    return rel


def get_activation(Z, activation='relu'):
    """
    Returns the results for a given activation.

    Parameters
    ----------
    Z : np.array
    activation : str
        String indicating the activation function to use. It can be
        'relu', 'sigmoid', 'tanh'
    
    Returns
    -------
    act : np.array
    """
    if activation=='relu':
        act = relu(Z)

    elif activation=='tanh':
        act = tanh(Z)
    
    elif activation=='sigmoid':
        act = sigmoid(Z)

    return act
