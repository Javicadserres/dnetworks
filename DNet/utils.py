import numpy as np


def initialize_parameters(input_dim, output_dim):
    """
    Initialize parameters
    """
    np.random.seed(1)
    den = np.sqrt(input_dim)

    weights = np.random.randn(output_dim, input_dim) / den
    bias = np.zeros((output_dim, 1))

    return weights, bias