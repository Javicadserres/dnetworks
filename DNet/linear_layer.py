import numpy as np
from activations import get_activation


def linear_forward(X, weights, bias):
    """
    Implement the linear part of a layer's forward propagation.

    Parameters
    ---------
    X : np.array
        Activations from previous layer or input data.
    weights : np.array
        weights matrix
    bias : np.array
        bias vector

    Returns
    -------
    Z : np.array
        The input of the activation function, also called 
        pre-activation parameter.
    cache : tuple
        Tuple containing the inputs, weights and biases; stored for 
        computing the backward pass efficiently.
    """
    Z = weights.dot(X) + bias
    cache = (X, weights, bias)
    
    return Z, cache


def linear_activation_forward(X, weights, bias, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION 
    layer.

    Parameters
    ----------
    X : np.array
        Activations from previous layer (or input data).
    weights : np.array
        Weights matrix.
    bias : np.array
        Bias vector.
    activation : str
        The activation to be used in this layer ("sigmoid", "tanh" 
        or "relu").

    Returns
    -------
    out_activation : np.array
        The output of the activation function, also called the 
        post-activation value.
    cache : tuple
        Tuple containing the "linear_cache" and "out_linear";
        stored for computing the backward pass efficiently.
    """
    out_linear, linear_cache = linear_forward(X, weights, bias)
    out_activation = get_activation(out_linear, activation)
    
    cache = (linear_cache, out_linear)

    return out_activation, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the deep learning model.
    
    Parameters
    ----------
    X : np.array
        Input variables.
    parameters : dict
        Dictionary containing your parameters "Wl", "bl":
        Wl : weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl : bias vector of shape (layer_dims[l], 1)
    
    Returns
    -------
    Y_hat : np.array 
        Last post-activation values.
    caches : list
        List of caches containing every cache of 
        linear_activation_forward().
    """
    caches = []
    num_layers = len(parameters) // 2      
    
    for layer in range(1, num_layers):
        X_prev = X
        weights = parameters['weights'+str(layer)]
        bias = parameters['bias'+str(layer)]

        X, cache = linear_activation_forward(
            X_prev, weights, bias, activation= 'relu'
        )
        caches.append(cache)
    
    weights = parameters['weights'+str(num_layers)]
    bias = parameters['bias'+str(num_layers)]

    Y_hat, cache = linear_activation_forward(
        X, weights, bias, activation= 'sigmoid'
    )
    caches.append(cache)
            
    return Y_hat, caches