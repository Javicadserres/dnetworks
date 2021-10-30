import numpy as np
import dnetworks

from dnetworks.layers import (
    ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
)

def test_relu():
    """
    Tests ReLU activation class.
    """
    Z = np.array([1, -1, 0])
    dA = np.array([2, 3, 5])

    expected_A = np.array([1, 0, 0])
    expected_dZ = np.array([2, 0, 0])

    activation = ReLU()
    obtained_A = activation.forward(Z)
    obtained_dZ = activation.backward(dA)

    np.testing.assert_almost_equal(expected_A, obtained_A)
    np.testing.assert_almost_equal(expected_dZ, obtained_dZ)


def test_sigmoid():
    """
    Tests Sigmoid activation class.
    """
    Z = np.array([1, -1, 0])
    dA = np.array([2, 3, 5])

    expected_A = np.array([0.73105858, 0.26894142, 0.5])
    expected_dZ = np.array([0.39322387, 0.5898358 , 1.25])

    activation = Sigmoid()
    obtained_A = activation.forward(Z)
    obtained_dZ = activation.backward(dA)

    np.testing.assert_almost_equal(expected_A, obtained_A)
    np.testing.assert_almost_equal(expected_dZ, obtained_dZ)

    
def test_tanh():
    """
    Tests hyperbolic tangent activation class.
    """
    Z = np.array([1, -1, -1])
    dA = np.array([2, 3, 5])

    expected_A = np.array([0.76159416, -0.76159416, -0.76159416])
    expected_dZ = np.array([0.83994868, 1.25992302, 2.09987171])

    activation = Tanh()
    obtained_A = activation.forward(Z)
    obtained_dZ = activation.backward(dA)

    np.testing.assert_almost_equal(expected_A, obtained_A)
    np.testing.assert_almost_equal(expected_dZ, obtained_dZ)


def test_leakyrelu():
    """
    Tests Leaky ReLU activation class.
    """
    Z = np.array([1, -1, 0])
    dA = np.array([2, 3, 5])

    expected_A = np.array([1, -0.01, 0])
    expected_dZ = np.array([2, 0.03, 0.05])

    activation = LeakyReLU()
    obtained_A = activation.forward(Z)
    obtained_dZ = activation.backward(dA)

    np.testing.assert_almost_equal(expected_A, obtained_A)
    np.testing.assert_almost_equal(expected_dZ, obtained_dZ)


def test_softmax():
    """
    Tests Softmax activation class.
    """
    Z = np.array([[1], [0], [0]])
    dA = np.array([[2], [3], [5]])

    expected_A = np.array([[0.57611688], [0.21194156], [0.21194156]])
    expected_dZ = np.array([[-0.48841244], [0.03226466], [0.45614778]])

    activation = Softmax()
    obtained_A = activation.forward(Z)
    obtained_dZ = activation.backward(dA)

    np.testing.assert_almost_equal(expected_A, obtained_A)
    np.testing.assert_almost_equal(expected_dZ, obtained_dZ)