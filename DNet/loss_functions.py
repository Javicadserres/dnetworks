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

    sub1 = np.dot(Y, np.log(Y_hat).T)
    sub2 = np.dot(1 - Y, np.log(1 - Y_hat).T)

    cost = (1./m) * (sub1 + sub2)

    return np.squeeze( - cost)


def mse_loss(Y_hat, Y):
    """
    Computes the mean squared error loss.

    Parameters
    ----------
    Y_hat : np.array
        Estimated values of the real Y.
    Y : np.array
        The lables.
    
    Returns
    -------
    mse : np.array
        The cost from the MSE loss.
    """
    sqrt_error = np.sqrt(Y_hat - Y)
    mse = np.mean(sqrt_error)

    return mse


def mae_loss(Y_hat, Y):
    """
    Computes the mean absolute error loss.

    Parameters
    ----------
    Y_hat : np.array
        Estimated values of the real Y.
    Y : np.array
        The lables.
    
    Returns
    -------
    mae : np.array
        The cost from the MAE loss.
    """
    abs_error = np.abs(Y_hat - Y)
    mae = np.mean(abs_error)

    return mae


def compute_loss(Y_hat, Y, cost_function):
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