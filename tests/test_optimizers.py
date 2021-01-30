import numpy as np
from DNet.optimizers import SGD, RMSprop, Adam
from tests.test_utils import test_parameters_optimizers


def test_sgd():
    """
    Tests Stochastic gradient descent optimization class.
    """
    weights, bias, dW, db = test_parameters_optimizers()   
    
    expected_weights = np.array(
        [[-0.17310385, -0.87734562,  0.04230592],
        [ 0.58351704, -1.10041826,  1.14432594]]
    )
    expected_bias = np.array([[0.90210947], [0.50279191]])
    expected_vdw = np.array(
        [[ 0.09008559, -0.06837279, -0.01228902],
        [-0.09357694, -0.02678881,  0.05303555]]
    )
    expected_vdb = np.array([[-0.06916607], [-0.03967535]])

    optimicer = SGD()
    (
        obtained_weigths, 
        obtained_bias, 
        (obtained_V_dW, obtained_V_db)
    ) = optimicer.optim(weights, bias, dW, db)

    np.testing.assert_almost_equal(expected_weights, obtained_weigths)
    np.testing.assert_almost_equal(expected_bias, obtained_bias)
    np.testing.assert_almost_equal(expected_vdw, obtained_V_dW)
    np.testing.assert_almost_equal(expected_vdb, obtained_V_db)


def test_rmsprop():
    """
    Tests RMSpropo optimization class.
    """
    weights, bias, dW, db = test_parameters_optimizers()   
    
    expected_weights = np.array(
        [[-0.19614529, -0.85414134,  0.06593083],
       [ 0.60653229, -1.0769021 ,  1.12100663]]
    )
    expected_bias = np.array([[0.9253078], [0.52621142]])
    expected_sdw = np.array(
        [[0.08115414, 0.04674838, 0.0015102],
       [0.08756644, 0.0071764 , 0.02812769]]
    )
    expected_sdb = np.array([[0.04783946], [0.01574134]])

    optimicer = RMSprop()
    (
        obtained_weigths, 
        obtained_bias, 
        (obtained_S_dW, obtained_S_db)
    ) = optimicer.optim(weights, bias, dW, db)

    np.testing.assert_almost_equal(expected_weights, obtained_weigths)
    np.testing.assert_almost_equal(expected_bias, obtained_bias)
    np.testing.assert_almost_equal(expected_sdw, obtained_S_dW)
    np.testing.assert_almost_equal(expected_sdb, obtained_S_db)


def test_adam():
    """
    Tests Adam optimization class.
    """
    weights, bias, dW, db = test_parameters_optimizers()   
    
    expected_weights = np.array(
        [[-0.17799666, -0.87228997, 0.0477822 ],
       [ 0.58838366, -1.09505073, 1.13915526]]
    )
    expected_bias = np.array([[0.90715917], [0.50806279]])
    expected_vdw = np.array(
        [[ 0.09008559, -0.06837279, -0.01228902],
       [-0.09357694, -0.02678881,  0.05303555]]
    )
    expected_vdb = np.array([[-0.06916607], [-0.03967535]])
    expected_sdw = np.array(
        [[0.00811541, 0.00467484, 0.00015102],
       [0.00875664, 0.00071764, 0.00281277]]
    )
    expected_sdb = np.array(
        [[0.00478395],
       [0.00157413]]
    )

    optimicer = Adam()
    (
        obtained_weigths, 
        obtained_bias, 
        (obtained_V_dW, obtained_V_db, obtained_S_dW, obtained_S_db, epoch)
    ) = optimicer.optim(weights, bias, dW, db, vel_square=(0, 0, 0, 0, 2))

    np.testing.assert_almost_equal(expected_weights, obtained_weigths)
    np.testing.assert_almost_equal(expected_bias, obtained_bias)
    np.testing.assert_almost_equal(expected_vdw, obtained_V_dW)
    np.testing.assert_almost_equal(expected_vdb, obtained_V_db)
    np.testing.assert_almost_equal(expected_sdw, obtained_S_dW)
    np.testing.assert_almost_equal(expected_sdb, obtained_S_db)