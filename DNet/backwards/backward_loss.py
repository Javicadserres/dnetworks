import numpy as np


def bce_backward(Y_hat, Y):
    """
    Implementes the backward propagation for the binary cross entropy
    funciton. 

    Parameters
    ----------
    Y_hat : np.array
        Predicted label.
    Y : np.array
        True "label".
    
    Returns
    -------
    grad : np.array
        Gradient of the cost with respect to the loss function same
        shape as Y_hat.
    """
    grad = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    return grad

    
def loss_backward(Y_hat, Y, loss_function):
    """
    Implement the backward propagation for a given loss function.
    
    Parameters
    ----------
    Y_hat : np.array
        Predicted label.
    Y : np.array
        True "label".
    loss_function : str
        The loss to be used in the deep neural network, it can be: 
        "BCE", "MAE" or "MSE".
    
    Returns
    -------
    grad_Y_hat : np.array
        Gradient of the cost with respect to the loss function same 
        shape as Y_hat.
    """
    Y = Y.reshape(Y_hat.shape)

    if loss_function=='BCE':
        grad_Y_hat = bce_backward(Y_hat, Y)

    return grad_Y_hat