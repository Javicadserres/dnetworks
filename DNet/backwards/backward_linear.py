import numpy as np


def linear_backward(grad_Z, cache):
    """
    Implement the linear portion of backward propagation for a single
    layer.

    Parameters
    ----------
    grad_Z : np.array
        Gradient of the cost with respect to the linear output (of 
        current layer l).
    cache : tuple
        Tuple of values (X_prev, weights, bias) coming from the 
        forward propagation in the current layer.

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
    X_prev, weights, _ = cache
    m = X_prev.shape[1]

    grad_weights = (1 / m) * grad_Z.dot(X_prev.T)
    grad_bias = np.mean(grad_Z, axis=1, keepdims=True)
    grad_X_prev = weights.T.dot(grad_Z)
    
    return grad_X_prev, grad_weights, grad_bias