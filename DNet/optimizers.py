import numpy as np


def optimize(parameters, grads, lr, method='SGD'):
    """
    Update parameters using a given optimization funciton.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the parameters to be optimized.
    grads : dict
        Dictionary containing the gradients of the parameters.
    lr : int
        Learning rate.
    method : str, default='SGD'
        Optimization function to be used, by default is the 
        stochastic gradient descent.
    
    Returns
    -------
    parameters : dict
        Dictionary containing the updated parameters. 
    """
    if method=='SGD':
        parameters = optimize_sgd(parameters, grads, lr)

    return parameters


def optimize_sgd(parameters, grads, lr):
    """
    Update parameters using stochastic gradient descent.
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing the parameters to be optimized.
    grads : dict
        Dictionary containing the gradients of the parameters.
    lr : int
        Learning rate.
    
    Returns
    -------
    parameters : dict
        Dictionary containing the updated parameters.
    """
    num_layers = len(parameters) // 2 

    for layer in range(num_layers):

        layer_str = str(layer + 1)
        weights = parameters["weights" + layer_str]
        bias = parameters["bias" + layer_str]

        parameters["weights" + layer_str] = (
            weights - lr * grads["grad_weights" + layer_str]
        )
        parameters["bias" + layer_str] = (
            bias - lr * grads["grad_bias" + layer_str]
        )

    return parameters