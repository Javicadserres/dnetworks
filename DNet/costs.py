import numpy as np 


def bce_loss(Y_hat, Y):
    """
    Computes the binary cross entropy loss.

    Parameters
    ----------
    Y_hat : np.array
        Estimated values of the real Y.
    Y : np.array
        The lables.
    
    Returns
    -------
    cost : np.array
        The cost from the BCE loss.
    """
    m = len(Y)

    sub1 = - np.dot(Y, np.log(Y_hat).T)
    sub2 = - np.dot(1 - Y, np.log(1 - Y_hat).T)

    cost = (1./m) * (sub1 + sub2)

    return np.squeeze(cost)