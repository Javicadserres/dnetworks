import numpy as np


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
