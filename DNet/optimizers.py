import numpy as np
from utils import compute_opt_update


class SGD:
    """
    Class that implements the gradient descent algorithm.
    """
    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int
            Learing rate to use for the gradient descent.
        beta : int
            Beta parameter.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weigths, bias, dW, db, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        velocities : tuple
            Tuple containing the velocities to compute the gradient
            descent with momentum.

        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        """
        if velocities is None:
            V_dW, V_db = (0, 0)
        else:
            V_dW, V_db = velocities

        V_dW = compute_opt_update(dW, self.beta, V_dW)
        V_db = compute_opt_update(db, self.beta, V_db)

        weigths -= self.lr * V_dW
        bias -= bias - self.lr * V_db

        velocities = (V_dW, V_db)

        return weigths, bias, velocities


class RMSprop:
    """
    Class that implements the RMSprop algorithm.
    """
    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int
            Learing rate to use for the gradient descent.
        beta : int
            Beta parameter to compute the sqaures.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weigths, bias, dW, db, squares=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        squares : tuple
            Tuple containing the square to compute the gradient
            descent with momentum.

        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        squares : tuple
            Tuple containing the square to compute the gradient
            descent with momentum.
        """
        if squares is None:
            S_dW, S_db = (0, 0)
        else:
            S_dW, S_db = squares

        S_dW = compute_opt_update(np.power(dW, 2), self.beta, S_dW)
        S_db = compute_opt_update(np.power(db, 2), self.beta, S_db)

        weigths -= self.lr * dW / np.sqrt(S_dW)
        bias -= self.lr * db / np.sqrt(S_db)

        squares = (S_dW, S_db)

        return weigths, bias, squares


class Adam:
    """
    Class that implements the Adaptive moment stimation algorithm.
    """
    def __init__(self, lr=0.0075, betas=(0.9, 0.99), epsilon=1e-18):
        """
        Parameters
        ----------
        lr : int
            Learing rate to use for the gradient descent.
        betas : tuple of int
            Betas parameters.
        epsilon : int
        """
        self.betas = betas
        self.lr = lr
        self.epsilon = epsilon


    def optim(self, weigths, bias, dW, db, vel_square=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        vel_square : tuple
            Tuple containing the square to compute the gradient
            descent with momentum.

        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        """
        if vel_square is None:
            V_dW, V_db, S_dW, S_db, epoch = (0, 0, 0, 0, 1)
        else:
            V_dW, V_db, S_dW, S_db, epoch = vel_square
        
        beta1, beta2 = self.betas

        V_dW = compute_opt_update(dW, beta1, V_dW)
        V_dW_c = V_dW / (1 - np.power(beta1, epoch))

        V_db = compute_opt_update(db, beta1, V_db)
        V_db_c = V_db / (1 - np.power(beta1, epoch))

        S_dW = compute_opt_update(np.power(dW, 2), beta2, S_dW)
        S_dW_c = S_dW / (1 - np.power(beta2, epoch))

        S_db = compute_opt_update(np.power(db, 2), beta2, S_db)
        S_db_c = S_db / (1 - np.power(beta2, epoch))

        weigths -= self.lr * V_dW_c / (np.sqrt(S_dW_c) + self.epsilon)
        bias -= self.lr * V_db_c / (np.sqrt(S_db_c) + self.epsilon)

        epoch += 1

        vel_square = (V_dW, V_db, S_dW, S_db, epoch)

        return weigths, bias, vel_square