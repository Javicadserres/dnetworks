import numpy as np
from activations import get_activation
from backwards import sigmoid_backward, relu_backward, bce_backward
from costs import bce_loss


def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters of the deep neural network.

    Parameters
    ---------
    layer_dims : list
        List containing the dimensions of each layer in our network
    
    Returns
    -------
    parameters : dict
        Dictionary containing your parameters "weightsl", "biasl":
        Wl : weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl : bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    parameters = {}
    num_layers = len(layer_dims)    

    for layer in range(1, num_layers):
        parameters['weights' + str(layer)] = np.random.randn(
            layer_dims[layer], layer_dims[layer - 1]
        ) / np.sqrt(layer_dims[layer - 1])
        parameters['bias' + str(layer)] = np.zeros(
            (layer_dims[layer], 1)
        )

    return parameters


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


def compute_cost(Y_hat, Y, cost_function):
    """
    Implement the defined cost function.

    Parameters
    ----------
    Y_hat : np.array
        Predicted label
    Y : np.array
        True "label"
    cost_function : str
        The cost function to be used. It can be 'BCE', 'MSE', 'MAE'.

    Returns
    -------
    cost : int
        Defined cost
    """
    if cost_function=='BCE':
        cost = bce_loss(Y_hat, Y)

    elif cost_function=='MSE':
        cost = mse_loss(Y_hat, Y)

    elif cost_function=='MAE':
        cost = mae_loss(Y_hat, Y)

    return cost


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
    X_prev, weights, bias = cache
    m = X_prev.shape[1]

    grad_weights = (1 / m) * grad_Z.dot(X_prev.T)
    grad_bias = np.mean(grad_Z, axis=1, keepdims=True)
    grad_X_prev = weights.T.dot(grad_Z)
    
    return grad_X_prev, grad_weights, grad_bias


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

