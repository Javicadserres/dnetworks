import numpy as np
import dnet

from dnet.layers import (
    BCELoss,
    MSELoss,
    MAELoss,
    CrossEntropyLoss
)


def test_bceloss():
    """
    Tests Binary Cross Entropy loss class.
    """
    Y_hat = np.array([1, 0.5, 0, 1, 0])
    Y = np.array([1, 1, 0, 1, 0])

    expected_loss = 0.13862943611198777
    expected_grad = np.array([-1., -2., 1., -1.,  1.])

    loss = BCELoss()
    obtained_loss = loss.forward(Y_hat, Y)
    obtained_grad = loss.backward()

    np.testing.assert_almost_equal(expected_loss, obtained_loss)
    np.testing.assert_almost_equal(expected_grad, obtained_grad)


def test_mseloss():
    """
    Tests Mean squared error loss class.
    """
    Y_hat = np.array([1, 0.5, 0, 1, 0])
    Y = np.array([1, 1, 0, 1, 0])

    expected_loss = 0.05
    expected_grad = np.array([0., -1., 0., 0., 0.])

    loss = MSELoss()
    obtained_loss = loss.forward(Y_hat, Y)
    obtained_grad = loss.backward()

    np.testing.assert_almost_equal(expected_loss, obtained_loss)
    np.testing.assert_almost_equal(expected_grad, obtained_grad)


def test_maeloss():
    """
    Tests Mean absolute error loss class.
    """
    Y_hat = np.array([1, 0.5, 0, 1, 0])
    Y = np.array([1, 1, 0, 1, 0])

    expected_loss = 0.1
    expected_grad = np.array([0., -1., 0., 0., 0.])

    loss = MAELoss()
    obtained_loss = loss.forward(Y_hat, Y)
    obtained_grad = loss.backward()

    np.testing.assert_almost_equal(expected_loss, obtained_loss)
    np.testing.assert_almost_equal(expected_grad, obtained_grad)


def test_crossentropyloss():
    """
    Tests Cross entropy loss class.
    """
    Y_hat = np.array([9, 3, 1, 1, 3])
    Y = np.array([1, 0, 0, 0, 0])

    expected_loss = 0.0056126491841890945
    expected_grad = np.array(
        [-0.00559693, 0.00246488, 0.00033359, 0.00033359, 0.00246488]
    )

    loss = CrossEntropyLoss()
    obtained_loss = loss.forward(Y_hat, Y)
    obtained_grad = loss.backward()

    np.testing.assert_almost_equal(expected_loss, obtained_loss)
    np.testing.assert_almost_equal(expected_grad, obtained_grad)