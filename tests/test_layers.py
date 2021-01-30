import numpy as np

from DNet.layers.linear_layers import LinearLayer
from DNet.layers.padding_layers import ConstantPad
from DNet.layers.convolutional_layers import Conv2D
from DNet.layers.pooling_layers import MaxPooling2D, AveragePooling2D
from tests.test_utils import (
    test_parameters_linearlayer, test_parameters_convolution
)


def test_linearlayer():
    """
    Tests the linear layer class.
    """
    weights, bias, A, dZ = test_parameters_linearlayer()

    layer = LinearLayer(3, 2)
    layer.weights = weights
    layer.bias = bias

    expected_Z = np.array([[-13.], [50.5]])
    expected_dA = np.array([[4.5],[3.], [7.6]])

    obtained_Z = layer.forward(A)
    obtained_dA = layer.backward(dZ)

    np.testing.assert_almost_equal(expected_Z, obtained_Z)
    np.testing.assert_almost_equal(expected_dA, obtained_dA)


def test_paddinglayer():
    """
    Tests padding layer class.
    """
    X = np.array([[1, 1, 1], [1, 1, 1]])
    
    expected = np.array(
        [[0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]]
    )
    padding = 1
    dimension = 2
    constant = 0
    padclass = ConstantPad(padding, dimension, constant)
    obtained = padclass.pad(X)

    np.testing.assert_almost_equal(expected, obtained)


def test_conv2d():
    """
    Test Convolutional 2D layer class.
    """
    image, weights, bias = test_parameters_convolution()

    expected_Z = np.array(
        [[[[17]], [[26]], [[11]], [[ 5]]],
        [[[39]], [[32]], [[28]], [[ 8]]],
        [[[29]], [[46]], [[24]], [[17]]],
        [[[43]], [[49]], [[50]], [[29]]]]
    )
    expected_dA = np.array(
        [[[[326.]], [[475.]], [[296.]], [[159.]]],
        [[[488.]], [[555.]], [[491.]], [[201.]]],
        [[[522.]], [[798.]], [[628.]], [[378.]]],
        [[[316.]], [[392.]], [[370.]], [[190.]]]]
    )
    expected_dW = np.array(
        [[[[771, 927, 626], [833, 1200, 739], [529, 749, 569]]]]
    )
    expected_db = np.array([[453]])

    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    
    convolution = Conv2D(
        in_channels, out_channels, kernel_size, stride, padding
    )
    convolution.weights = weights
    convolution.bias = bias

    obtained_Z = convolution.forward(image)
    obtained_dA = convolution.backward(obtained_Z)
    obtained_dW = convolution.dW
    obtained_db = convolution.db

    np.testing.assert_almost_equal(expected_Z, obtained_Z)
    np.testing.assert_almost_equal(expected_dA, obtained_dA)
    np.testing.assert_almost_equal(expected_dW, obtained_dW)
    np.testing.assert_almost_equal(expected_db, obtained_db)


def test_maxpooling2d():
    """
    Tests Max pooling layer class.
    """
    image, _, _ = test_parameters_convolution()

    expected_Z = np.array(
        [[[[4]], [[4]]], [[[4]], [[4]]]]
    )
    expected_dA = np.array(
        [[[[ 0.]], [[ 8.]], [[ 0.]], [[ 0.]]],
        [[[ 0.]], [[ 0.]], [[ 0.]], [[ 0.]]],
        [[[ 8.]], [[16.]], [[ 0.]], [[ 0.]]],
        [[[ 4.]], [[ 0.]], [[ 8.]], [[ 0.]]]]
    )

    kernel_size = (3, 3)
    stride = 1
    padding = 0

    convolution = MaxPooling2D(kernel_size, stride, padding)

    obtained_Z = convolution.forward(image)
    obtained_dA = convolution.backward(obtained_Z)

    np.testing.assert_almost_equal(expected_Z, obtained_Z)
    np.testing.assert_almost_equal(expected_dA, obtained_dA)


def test_averagepooling2d():
    """
    Tests Max pooling layer class.
    """
    image, _, _ = test_parameters_convolution()

    expected_Z = np.array([[[[2.11111111]]]])
    expected_dA = np.array(
        [[[[0.2345679]], [[0.2345679]], [[0.2345679]], [[0.]]],
        [[[0.2345679]], [[0.2345679]], [[0.2345679]], [[0.]]],
        [[[0.2345679]], [[0.2345679]], [[0.2345679]], [[0.]]],
        [[[0.]], [[0.]], [[0.]], [[0.]]]]
    )

    kernel_size = (3, 3)
    stride = 2
    padding = 0

    convolution = AveragePooling2D(kernel_size, stride, padding)

    obtained_Z = convolution.forward(image)
    obtained_dA = convolution.backward(obtained_Z)

    np.testing.assert_almost_equal(expected_Z, obtained_Z)
    np.testing.assert_almost_equal(expected_dA, obtained_dA)