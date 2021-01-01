import numpy as np
from backwards.backward_activation import (
    relu_backward, sigmoid_backward
)
from backwards.backward_linear import  linear_backward
from backwards.backward_loss import loss_backward


def linear_activation_backward(grad_X, cache, activation):
    """
    Implement the backward propagation for a given layer.
    
    Parameters
    ----------
    grad_X : np.array
        Post-activation gradient for current layer.
    cache : tuple
        Tuple of values (linear_cache, out_linear) we store for 
        computing the backward propagation.
    activation : str
        The activation to be used in this layer, it can be: 
        "sigmoid", "relu" or "tanh".
    
    Returns
    -------
    grad_X_prev : np.array
        Gradient of the cost with respect to the activation (of the 
        previous layer), same shape as X_prev.
    grad_weights : np.array
        Gradient of the cost with respect to the weights (current 
        layer), same shape as weights.
    grad_bias : np.array
        Gradient of the cost with respect to the bias (current layer)
        same shape as bias.
    """
    linear_cache, out_linear = cache
    
    if activation == "relu":
        grad_Z = relu_backward(grad_X, out_linear)
        
    elif activation == "sigmoid":
        grad_Z = sigmoid_backward(grad_X, out_linear)
    
    return linear_backward(grad_Z, linear_cache)


def L_model_backward(Y_hat, Y, caches):
    """
    Implement the backward propagation for the deep learning model.
    
    Parameters
    ----------
    Y_hat : np.array
        Predicted label.
    Y : np.array
        True "label".
    caches : list
        List of caches containing every cache of 
        linear_activation_forward().
    
    Returns
    -------
    grads : dict
        A dictionary containing the gradients of the X, weights and
        bias variables.
    """
    grads = {}
    grad_Y_hat = loss_backward(Y_hat, Y, loss_function='BCE')

    num_layers = len(caches) 
    layers = range(num_layers)

    for layer in reversed(layers):

        if layer==(num_layers-1):
            activation = 'sigmoid'
            grad_X_prev = grad_Y_hat

        else:
            activation = 'relu'
            grad_X_prev = grads["grad_X" + str(layer + 1)]

        current_cache = caches[layer]
        (
            grads["grad_X" + str(layer)], 
            grads["grad_weights" + str(layer + 1)], 
            grads["grad_bias" + str(layer + 1)]
        ) = linear_activation_backward(
            grad_X_prev, current_cache, activation
        )

    return grads