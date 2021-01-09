import numpy as np


def initialize_parameters(input_dim, output_dim):
    """
    Initialize parameters randomly. The weights are divided by the 
    square root of the input dimensions in order to partially solve
    the vanishig and exploding gradient descents problem.

    Parameters
    ----------
    input_dim : int
        Dimension of the inputs.
    output_dim : int
        Dimensions of the output.

    Returns
    -------
    weights : numpy.array
        Array containing the initalize weights.
    bias : numpy.array
        Array containing zeros for the bias.
    """
    np.random.seed(1)
    den = np.sqrt(input_dim)

    weights = np.random.randn(output_dim, input_dim) / den
    bias = np.zeros((output_dim, 1))

    return weights, bias