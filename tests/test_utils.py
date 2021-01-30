import numpy as np


def test_parameters_optimizers():
    """
    Initialize paramaters to be used in the tests.

    Returns
    -------
    weights : numpy.array
    bias : numpy.array
    dW : numpy.array
    db : numpy.array
    """
    weights = np.array(
        [[-0.17242821, -0.87785842,  0.04221375],
       [ 0.58281521, -1.10061918,  1.14472371]]
    )
    bias = np.array([[0.90159072], [0.50249434]])

    dW = np.array(
        [[ 0.90085595, -0.68372786, -0.12289023],
       [-0.93576943, -0.26788808,  0.53035547]]
    )
    db = np.array([[-0.69166075], [-0.39675353]])
    
    return weights, bias, dW, db


def test_parameters_linearlayer():
    """
    Initialize paramaters to be used in the tests.

    Returns
    -------
    weights : numpy.array
    bias : numpy.array
    A : numpy.array
    dZ : numpy.array
    """
    weights = np.array([[-0.5, -1, 0.4], [2, 1, 4]])
    bias = np.array([[0.5], [0.5]])

    A = np.array([[-5], [20], [10]])
    dZ = np.array([[-1], [2]])

    return weights, bias, A, dZ


def test_parameters_convolution():
    """
    Initialize parameters to be used in the test.
    """
    image = np.array(
        [[[[3]], [[4]], [[0]], [[1]]],
        [[[3]], [[0]], [[0]], [[1]]],
        [[[4]], [[4]], [[1]], [[2]]],
        [[[4]], [[2]], [[4]], [[3]]]]
    )
    weights = np.array([[[[4, 2, 1], [2, 4, 0], [4, 1, 1]]]])
    bias = np.array([[1]])

    return image, weights, bias